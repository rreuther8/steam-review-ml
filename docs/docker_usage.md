# Docker usage (WSL)

Environment setup and CLI reference for Docker on this machine (WSL2 Ubuntu 24.04, systemd enabled). Source: [Docker CLI cheat sheet](https://docs.docker.com/get-started/docker_cheatsheet.pdf).

## Daemon lifecycle

Docker Engine is installed natively in this WSL distro (not Docker Desktop) via the official apt repo. The daemon does **not** auto-start on terminal launch — start it manually each session:

```bash
sudo systemctl start docker.socket docker.service
```

Stop it (both units — `docker.socket` re-spawns `docker.service` on the next connection otherwise):

```bash
sudo systemctl stop docker.socket docker.service
```

Shortcuts in `~/.bash_aliases`: `dockerup` / `dockerdown`.

Check status:

```bash
systemctl is-active docker.socket docker.service   # "active" or "inactive"
docker ps                                          # confirms the daemon is actually reachable
```

## Images

| Action | Command |
|---|---|
| Build from Dockerfile in current dir | `docker build -t <image_name> .` |
| Build without cache | `docker build -t <image_name> . --no-cache` |
| List local images | `docker images` |
| Delete an image | `docker rmi <image_name>` |
| Remove all unused images | `docker image prune` |
| Search Docker Hub | `docker search <image_name>` |
| Pull from Docker Hub | `docker pull <image_name>` |
| Login to Docker Hub | `docker login -u <username>` |
| Push to Docker Hub | `docker push <username>/<image_name>` |

## Containers

| Action | Command |
|---|---|
| Run a container, custom name | `docker run --name <container_name> <image_name>` |
| Run + publish a port | `docker run -p <host_port>:<container_port> <image_name>` |
| Run in the background (detached) | `docker run -d <image_name>` |
| Start / stop an existing container | `docker start\|stop <container_name>` |
| Remove a stopped container | `docker rm <container_name>` |
| Shell into a running container | `docker exec -it <container_name> sh` |
| Follow logs | `docker logs -f <container_name>` |
| Inspect a container | `docker inspect <container_name>` |
| List running containers | `docker ps` |
| List all containers (running + stopped) | `docker ps --all` |
| Resource usage stats | `docker container stats` |

## General

| Action | Command |
|---|---|
| Help (any subcommand takes `--help`) | `docker --help` |
| System-wide info (also confirms daemon reachable) | `docker info` |

## Why native Engine instead of Docker Desktop

Chose Engine installed directly in WSL over Docker Desktop's WSL2 backend: no separate Windows-side app to manage, and it forces daemon/service management (systemd units, group permissions) through the same path used on any Linux host (EC2, GCP VM, bare metal) rather than hiding it behind a GUI toggle.
