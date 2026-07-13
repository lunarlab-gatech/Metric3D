if [ "$(docker inspect -f '{{.State.Running}}' metric3d 2>/dev/null)" != "true" ]; then
    docker start metric3d 2>/dev/null || (echo "Container not found. Run run_container.sh first." && exit 1)
fi

docker exec -it metric3d /bin/bash
