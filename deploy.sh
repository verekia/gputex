docker buildx build --platform linux/arm64 --load -t verekia/gputex .
docker save -o /tmp/gputex.tar verekia/gputex
scp /tmp/gputex.tar midgar:/tmp/
ssh midgar docker load --input /tmp/gputex.tar
ssh midgar docker compose up -d gputex
