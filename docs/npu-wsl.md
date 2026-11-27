# Build the A733 NPU model in WSL

Use Ubuntu 22.04 on x86-64 WSL 2 to convert the SmartCane PP-LiteSeg ONNX model.
The Cubie A7S runs the exported A733 `.nb` model.

## Start WSL and load the converter

Start Docker Desktop and enable **Settings > Resources > WSL Integration** for
Ubuntu 22.04. Run the following in PowerShell. [Docker Desktop WSL setup](https://docs.docker.com/desktop/features/wsl/)

```powershell
wsl -l -v
wsl -d Ubuntu-22.04
```

In Ubuntu, create the conversion workspace and check Docker.

```bash
mkdir -p ~/smartcane-npu/docker-sdk
cd ~/smartcane-npu
uname -m
docker version --format 'client={{.Client.Version}} server={{.Server.Version}}'
```

If `ubuntu-npu:v2.0.10.2` is already loaded, continue to the model setup.
For a fresh setup, follow the A733 download link in the Radxa guide and save
`docker_images_v2.0.x.zip` under `~/smartcane-npu/docker-sdk/`. [Radxa ACUITY setup and download](https://docs.radxa.com/en/cubie/a7s/app-dev/npu-dev/cubie-acuity-env)

```bash
unzip docker-sdk/docker_images_v2.0.x.zip -d docker-sdk
unzip docker-sdk/docker_images_v2.0.x/ubuntu-npu_v2.0.10.2.tar.zip -d docker-sdk
docker load -i docker-sdk/ubuntu-npu_v2.0.10.2.tar
docker image inspect ubuntu-npu:v2.0.10.2 --format '{{.Os}}/{{.Architecture}}'
```

The expected image platform is `linux/amd64`. The commands below call Pegasus
inside the Docker image. The Allwinner Model Zoo's `examples/ppseg/` provides
a reference conversion. [Model Zoo download](https://docs.radxa.com/en/cubie/a7s/app-dev/npu-dev/model-zoo/model-zoo-download) and
[Radxa PPSeg example](https://docs.radxa.com/en/cubie/a7s/app-dev/npu-dev/model-zoo/ppseg)

## Put the inputs in the WSL workspace

All WSL commands below assume `~/smartcane-npu` is the current directory.
Copy `models/cane_ppliteseg_640x352.onnx` from this Git checkout into
`models/` in the WSL workspace. If the checkout is on the Cubie, replace
`CUBIE_HOST` with its SSH hostname or IP address and copy the model from there.

```bash
cd ~/smartcane-npu
mkdir -p models npu-work/{quant-images,import,export}
cubie=radxa@CUBIE_HOST
scp "$cubie:meam5160-group-9-smartcane/models/cane_ppliteseg_640x352.onnx" \
  models/
sha256sum models/cane_ppliteseg_640x352.onnx
```

The source ONNX used for this export has SHA-256 `df0d5572a91a079e15153a4e5289928af1624069ef1b3a0fc4e97d8516ebbf75`.

Put raw JPG or PNG images from the cane camera directly in
`npu-work/quant-images/`. Keep their original dimensions, use filenames
without spaces, and include road, sidewalk, and obstacles under different
lighting. Keep some images outside that directory for CPU/NPU comparisons.

```text
~/smartcane-npu/
|-- docker-sdk/
|-- models/
|   `-- cane_ppliteseg_640x352.onnx
`-- npu-work/
    |-- quant-images/
    |-- import/
    `-- export/
```

Generate the image list using the paths that Pegasus will see inside the container.

```bash
find npu-work/quant-images -maxdepth 1 -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
  -printf '/samples/%f\n' | LC_ALL=C sort > npu-work/import/dataset.txt
wc -l npu-work/import/dataset.txt
```

## Start the conversion container

Run this from WSL. The mounts place the input under `/input`, images under
`/samples`, imported files under `/work`, and exports under `/out`.

```bash
docker run --rm --pull=never -it \
  -v "$PWD/models:/input:ro" \
  -v "$PWD/npu-work/quant-images:/samples:ro" \
  -v "$PWD/npu-work/import:/work" \
  -v "$PWD/npu-work/export:/out" \
  -w /work ubuntu-npu:v2.0.10.2 bash
```

## Import and quantize the model

Run the following in the container. The image used for this build has Pegasus at `/usr/local/acuity_command_line_tools/pegasus.py`.

```bash
set -e
model=cane_ppliteseg_640x352
pegasus=/usr/local/acuity_command_line_tools/pegasus.py

python3 "$pegasus" import onnx \
  --model "/input/$model.onnx" \
  --output-model "$model.json" --output-data "$model.data"

python3 "$pegasus" generate inputmeta \
  --model "$model.json" --separated-database \
  --input-meta-output "${model}_inputmeta.yml"
```

Copy the input metadata for quantization and keep the imported tensor layout.
The model takes RGB at `640 x 352` in NCHW order. Normalize pixel values with
`(pixel - 127.5) / 127.5`, matching the CPU runner. Keep preprocessing outside
the exported graph.

```bash
test -s dataset.txt
sample_count=$(wc -l < dataset.txt)

python3 - "$model" <<'PY'
import sys
from pathlib import Path

import yaml

model = sys.argv[1]
source = Path(f"{model}_inputmeta.yml")
target = Path(f"{model}_pcq_inputmeta.yml")
config = yaml.safe_load(source.read_text())
database = config["input_meta"]["databases"][0]
database["path"] = "/work/dataset.txt"
preprocess = database["ports"][0]["preprocess"]
preprocess["reverse_channel"] = False
preprocess["mean"] = [127.5] * 3
preprocess["scale"] = [1.0 / 127.5] * 3
preprocess["preproc_node_params"]["add_preproc_node"] = False
target.write_text(yaml.safe_dump(config, sort_keys=False))
PY

python3 "$pegasus" quantize \
  --model "$model.json" --model-data "$model.data" \
  --device CPU --with-input-meta "${model}_pcq_inputmeta.yml" \
  --iterations "$sample_count" --rebuild \
  --model-quantize "${model}_pcq.quantize" \
  --quantizer perchannel_symmetric_affine --qtype int8
```

A successful run writes `cane_ppliteseg_640x352_pcq.quantize` to
`npu-work/import/` and rebuilds the `.data` weights there. Use those rebuilt
weights for the export. [Radxa ACUITY quantization guide](https://docs.radxa.com/en/cubie/a7s/app-dev/npu-dev/cubie-acuity-usage#model-quantization)

## Export for the Cubie A7S

In the same container, use the A733 optimization target and pack the NBG binary. [A733 export parameters](https://doc.kickpi.com/products/linux_customization/linux_npu/)

```bash
viv_sdk=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
build_dir=$(mktemp -d /out/build.XXXXXX)
postprocess="$build_dir/postprocess.yml"

python3 "$pegasus" generate postprocess-file \
  --model "$model.json" --postprocess-file-output "$postprocess"

python3 "$pegasus" export ovxlib \
  --model "$model.json" --model-data "$model.data" \
  --dtype quantized --model-quantize "${model}_pcq.quantize" \
  --batch-size 1 --target-ide-project linux64 --save-fused-graph \
  --with-input-meta "${model}_pcq_inputmeta.yml" \
  --postprocess-file "$postprocess" \
  --optimize VIP9000NANODI_PLUS_PID0X1000003B \
  --viv-sdk "$viv_sdk" --pack-nbg-unify \
  --output-path "$build_dir/${model}_pcq_a733"

nb="${build_dir}_nbg_unify/network_binary.nb"
test -s "$nb"
cp "$nb" "/out/${model}_pcq_a733.nb"
exit
```

Pegasus adds `_nbg_unify` to the build directory name. The named export is at
`npu-work/export/cane_ppliteseg_640x352_pcq_a733.nb` in WSL.

## Copy the result to the Cubie

In WSL, copy the model to `models/` in the Cubie checkout. Use the same SSH
destination as above and adjust the checkout location if needed.

```bash
scp npu-work/export/cane_ppliteseg_640x352_pcq_a733.nb \
  "$cubie:meam5160-group-9-smartcane/models/"
```

On the Cubie, build the native library and run the live preview. For a new
export, compare CPU and NPU class masks on the same held-out camera images
before checking the live BEV. [NPU build and live preview instructions](../README.md#live-a733-npu-segmentation-and-bev)
