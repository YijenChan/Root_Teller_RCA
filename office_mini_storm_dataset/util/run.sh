# ./run.sh
# docker run -d --name=fluent-bit \
#   -v ./officestorm/data:/fluentbit \
#   -v ./fluent-bit.conf:/fluent-bit/etc/fluent-bit.conf \
#   -v ./json-parsers.conf:/fluent-bit/etc/json-parsers.conf \
# fluent/fluent-bit:latest

docker run -d --name=fluent-bit \
    -ti \
    -v ../data:/fluentbit \
    -v ./fluent-bit.yaml:/fluent-bit/etc/fluent-bit.yaml \
    -v ./json-parsers.conf:/fluent-bit/etc/json-parsers.conf \
    fluent/fluent-bit:latest \
    -c /fluent-bit/etc/fluent-bit.yaml 
