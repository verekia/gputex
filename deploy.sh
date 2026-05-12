docker buildx build --platform linux/arm64 --load -t verekia/gputex .
docker save verekia/gputex | gzip > /tmp/gputex.tar.gz
scp /tmp/gputex.tar.gz midgar:/tmp/
ssh midgar docker load --input /tmp/gputex.tar.gz
ssh midgar docker compose up -d gputex
