from collections.abc import Iterable, Iterator, Mapping
from os import PathLike
from typing import IO, Any, TypeAlias, Union, cast

from fsspec import AbstractFileSystem, url_to_fs
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem

from refiner.io.datafile import DataFile, _storage_options_for_path

DataFolderPath: TypeAlias = str | PathLike[str]
DataFolderSpec: TypeAlias = tuple[DataFolderPath, AbstractFileSystem]
DataFolderLike: TypeAlias = Union[DataFolderPath, DataFolderSpec, "DataFolder"]


class DataFolder(DirFileSystem):
    """A simple wrapper around fsspec's DirFileSystem to handle file listing and sharding files across multiple workers/process.
        Also handles the creation of output files.
        All file operations will be relative to `path`.

    Args:
        path: the path to the folder (local or remote)
        fs: the filesystem to use (see fsspec for more details)
        auto_mkdir: whether to automatically create the parent directories when opening a file in write mode
        **storage_options: additional options to pass to the filesystem
    """

    def __init__(
        self,
        path: str,
        fs: AbstractFileSystem | None = None,
        auto_mkdir: bool = True,
        **storage_options,
    ):
        """
        Objects can be initialized with `path`, `path` and `fs` or `path` and `storage_options`
        Args:
            path: main path to base directory
            fs: fsspec filesystem to wrap
            auto_mkdir: if True, when opening a file in write mode its parent directories will be automatically created
            **storage_options: will be passed to a new fsspec filesystem object, when it is created. Ignored if fs is given
        """
        super().__init__(
            path=path,
            fs=fs if fs is not None else url_to_fs(path, **storage_options)[0],
        )
        self.auto_mkdir = auto_mkdir

    @classmethod
    def resolve(
        cls,
        data: DataFolderLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
    ) -> "DataFolder":
        """
        `DataFolder` factory.
        Possible input combinations:
        - `str`: the simplest way is to pass a single string. Example: `/home/user/mydir`, `s3://mybucket/myinputdata`,
          `hf://datasets/allenai/c4/en/`
        - `DataFolder`: you can initialize a DataFolder object directly and pass it as an argument


        Args:
            data: `DataFolder` | `str` | `(path, fs)`
            fs: Optional initialized filesystem to use. If provided, `storage_options` is ignored.
            storage_options: Optional fsspec filesystem init options (used only when `fs` is not provided).

        Returns:
            `DataFolder` instance
        """
        # Like DataFile.resolve(), this normalizes a folder handle but does not list it.
        # fully initialized DataFolder object
        if isinstance(data, cls):
            return data
        if (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[1], AbstractFileSystem)
        ):
            spec = cast(DataFolderSpec, data)
            data = spec[0]
            fs = spec[1]
        if isinstance(data, PathLike):
            data = str(data)
        # simple string path
        if isinstance(data, str):
            if fs is not None:
                path = fs._strip_protocol(data)
                return cls(path, fs=fs)
            return cls(data, **_storage_options_for_path(data, storage_options))
        raise TypeError(
            "You must pass a DataFolder instance, str path, PathLike, or (path, fs)"
        )

    def abs_path(self, path: str = "") -> str:
        # make sure we strip file:// and similar
        return self.fs.unstrip_protocol(self._join(path)).removeprefix("file://")

    def abs_paths(self, paths: str | Iterable[str]) -> str | list[str]:
        """
        Transform  a list of relative paths into a list of complete paths (including fs protocol and base path)

        Args:
            paths: list of relative paths

        Returns:
            list of fully resolved paths

        """
        if isinstance(paths, str):
            return self.abs_path(paths)
        return [self.abs_path(p) for p in paths]

    def find(self, path: str, *args, **kwargs):
        # Avoid DirFileSystem.find(): some backends (notably HF buckets) can leak
        # sibling prefix matches like `root-2/...` or return the bare root entry,
        # and DirFileSystem._relpath() asserts before we can filter them out.
        """List paths under this folder, skipping backend results outside the base path."""
        detail = kwargs.get("detail", False)
        target = self._join(path.rstrip("/"))
        ret = self.fs.find(target, *args, **kwargs)
        target = target.rstrip("/")
        target_prefix = target + self.fs.sep
        alt_target = target[1:] if target.startswith(self.fs.sep) else None
        alt_prefix = alt_target + self.fs.sep if alt_target is not None else None

        def rel(p: str) -> str | None:
            if p == target or (alt_target is not None and p == alt_target):
                return path.rstrip("/")
            if p.startswith(target_prefix):
                suffix = p[len(target_prefix) :]
            elif alt_prefix is not None and p.startswith(alt_prefix):
                suffix = p[len(alt_prefix) :]
            else:
                return None
            return suffix if path in {"", "/"} else f"{path.rstrip('/')}/{suffix}"

        if detail:
            return {r: info for p, info in ret.items() if (r := rel(p)) is not None}
        return [r for p in ret if (r := rel(p)) is not None]

    def open_files(
        self, paths: Iterable[str], mode: str = "rb", **kwargs
    ) -> list[IO[Any]]:
        """Opens all files in an iterable with the given options, in the same order as given

        Args:
            paths: iterable of relative paths
            mode: the mode to open the files with (Default value = "rb")
            **kwargs: additional arguments to pass to the open

        Returns:
            list of opened files
        """
        return [self.open(path, mode=mode, **kwargs) for path in paths]

    def open(self, path, *args, **kwargs):
        """Open a file locally or remote, and create the parent directories if self.auto_mkdir is `True` and we are opening in write mode.

            args/kwargs will depend on the filesystem (see fsspec for more details)
            Typically we often use:
                - compression: the compression to use
                - block_size: the block size to use

        Args:
            path: the path to the file
            *args: additional arguments to pass to the open
            **kwargs: additional arguments to pass to the open
        """
        mode: str = kwargs.pop("mode", args[0] if args else "rb")
        if self.auto_mkdir and isinstance(mode, str) and (set(mode) & set("wax+")):
            self.fs.makedirs(self.fs._parent(self._join(path)), exist_ok=True)
        return super().open(path, *args, mode=mode, **kwargs)

    @property
    def is_local(self) -> bool:
        """
        Checks if the underlying fs instance is a LocalFileSystem
        """
        return isinstance(self.fs, LocalFileSystem)

    def file(self, relpath: str) -> DataFile:
        """Create a DataFile pointing to a file under this folder."""
        return DataFile(fs=self.fs, path=self._join(relpath))

    def files(self, relpaths: Iterable[str]) -> list[DataFile]:
        return [self.file(p) for p in relpaths]

    def iter_files_with_sizes(
        self, *, recursive: bool = False, **kwargs: Any
    ) -> Iterator[tuple[DataFile, int | None]]:
        if recursive:
            found = self.find("", detail=True, **kwargs)
            items: Iterable[tuple[str, Mapping[str, Any]]] = found.items()
        else:
            items = (
                (str(info["name"]), info)
                for info in self.ls("", detail=True, **kwargs)
                if isinstance(info, Mapping)
            )

        for relpath, info in sorted(items, key=lambda item: item[0]):
            info_dict = dict(info)
            if info_dict.get("type") != "file":
                continue
            size = info_dict.get("size")
            yield self.file(relpath), size if isinstance(size, int) else None

    def iter_files(
        self, *, recursive: bool = False, **kwargs: Any
    ) -> Iterator[DataFile]:
        for file, _ in self.iter_files_with_sizes(recursive=recursive, **kwargs):
            yield file
