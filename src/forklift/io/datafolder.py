from collections.abc import Mapping, Iterable
from fsspec import AbstractFileSystem, url_to_fs
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing import IO, TypeAlias, Any
from .datafile import DataFile

DataFolderLike: TypeAlias = (
    str | tuple[str, Mapping[str, Any]] | tuple[str, AbstractFileSystem] | "DataFolder"
)


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
    def resolve(cls, data: DataFolderLike) -> "DataFolder":
        """
        `DataFolder` factory.
        Possible input combinations:
        - `str`: the simplest way is to pass a single string. Example: `/home/user/mydir`, `s3://mybucket/myinputdata`,
        `hf://datasets/allenai/c4/en/`
        - `(str, fsspec filesystem instance)`: a string path and a fully initialized filesystem object.
        Example: `("s3://mybucket/myinputdata", S3FileSystem(client_kwargs={"endpoint_url": endpoint_uri}))`
        - `(str, Mapping)`: a string path and a dictionary with options to initialize a fs. Example
        (equivalent to the previous line): `("s3://mybucket/myinputdata", {"client_kwargs": {"endpoint_url": endpoint_uri}})`
        - `DataFolder`: you can initialize a DataFolder object directly and pass it as an argument


        Args:
            data: DataFolder | str | tuple[str, Mapping] | tuple[str, AbstractFileSystem]:

        Returns:
            `DataFolder` instance
        """
        # fully initialized DataFolder object
        if isinstance(data, cls):
            return data
        # simple string path
        if isinstance(data, str):
            return cls(data)
        # (str path, fs init options Mapping)
        if (
            isinstance(data, tuple)
            and isinstance(data[0], str)
            and isinstance(data[1], Mapping)
        ):
            return cls(data[0], **data[1])
        # (str path, initialized fs object)
        if (
            isinstance(data, tuple)
            and isinstance(data[0], str)
            and isinstance(data[1], AbstractFileSystem)
        ):
            return cls(data[0], fs=data[1])
        raise TypeError(
            "You must pass a DataFolder instance, a str path, a (str path, fs_init_kwargs) or (str path, fs object)"
        )

    def _abs_path(self, path: str) -> str:
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
            return self._abs_path(paths)
        return [self._abs_path(p) for p in paths]

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
        mode: str = kwargs.get("mode", args[0] if args else "rb")
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
