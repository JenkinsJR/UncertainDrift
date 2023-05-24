import itertools


def gen_numerical(directory, ext, start=0, mkdir=True):
    if mkdir:
        directory.mkdir(parents=True, exist_ok=True)
    for i in itertools.count(start=start):
        yield (directory / str(i)).with_suffix(ext)


def sort_numerical(paths):
    return sorted(paths, key=lambda x: int(x.stem))