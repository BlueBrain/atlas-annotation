# What is this?
DVC is a library for tracking data and pipelines (= stages) necessary
to reproduce it. Like in git, data and pipelines can be versioned
and stored on a remote server. More information here: https://dvc.org/

In this folder we track:
- Nissl volume (25 microns)
- Average brain volume (25 microns)
- CCFv2 annotation atlas (2011, with fibers)
- CCFv3 annotation atlas (2017)
- Dimitri's coarse and fine atlas merging stages
- The resulting merged CCFv2 and CCFv3 merged atlases

# How do I list all tracked files?
```shell
dvc list --dvc-only .
```

# How do I list all stages?
```shell
dvc stage list
```
To list all logical relations and dependencies between stages run

```shell
dvc dag
```

# How do I obtain a given data file?
In analogy to git, a tracked file can be pulled from the remote via
```shell
dvc pull <tracked file>
```
It's also possible to specify a stage name instead of a
tracked file. In this case the outputs of the stage will be pulled.

To pull all available
data files just drop the tracked file argument:
```shell
dvc pull
```

If a file is the result of a stage, then it can be recreated from
scratch via
```shell
dvc repro <stage name>
```

Depending on the stage this can take some time (especially the atlas
merging stages).

Finally, to reproduce everything run
```shell
dvc repro
```

Since all our tracked data is the output of some stage, this will
recreate all tracked data.
