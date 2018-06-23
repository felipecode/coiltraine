The current dataset style is hdf5 files read with h5py

This is interesting since is a more compact way to store uncompressed
images. H5py allows cheaper random access that is very useful to
be used with my global balancing strategy.
H5PY also allows extra data compression and you can choose your file
size.


Bug: the h5py library seems to be caching the data read, making some
memory explosion. This is only happening when using pytorch
multithreaded access.

TODO: is this the optimal solutioon ? Is there a better way to store data?
How should data be organized ??