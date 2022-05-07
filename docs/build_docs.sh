rm -rf build

# update api rst
#rm -rf source/api/
#sphinx-apidoc --module-first -o source/api/ ../easycv/
make html
cp  -r ../configs  build/
if [ ! -d build/benchmarks ]; then
    mkdir -p build/benchmarks/
fi
cp  -r ../benchmarks/selfsup  build/benchmarks/
