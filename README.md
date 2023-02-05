# Document Filtering

## Run with docker

### Build the image

```bash
docker build -t doc_search .
```

### Run the container in the bash shell

```bash
docker run -it -v $(pwd)/data/:/documentSearch/data doc_search /bin/bash
```

- `-it` specifies that you want to run the container in an interactive mode
- `-v` option is used to specify the volume mount. The first part of the volume mount, `$(pwd)/data/`, is the host
  directory that you want to mount. The second part, `/documentSearch/data`, is the container directory where the host
  directory will
  be mounted.
- the final argument, `/bin/bash`, specifies the command to run in the container. It allows you to run commands inside the
  container. You can exit the shell by typing `exit`.