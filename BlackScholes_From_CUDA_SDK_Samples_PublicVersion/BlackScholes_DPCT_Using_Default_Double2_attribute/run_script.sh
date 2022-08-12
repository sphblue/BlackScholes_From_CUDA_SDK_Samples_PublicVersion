set -x
#EXEC=BlackScholes.dpct.nvgpu.float.ortce-a100
#EXEC=BlackScholes.dpct.nvgpu.double.ortce-a100

EXEC=BlackScholes.dpct.nvgpu.float.nvrtx
#EXEC=BlackScholes.dpct.nvgpu.double.nvrtx

./${EXEC} -i:1000 -o:1024
./${EXEC} -i:1000 -o:1024
./${EXEC} -i:1000 -o:4096
./${EXEC} -i:1000 -o:4096
./${EXEC} -i:1000 -o:8192
./${EXEC} -i:1000 -o:8192
./${EXEC} -i:1000 -o:81920
./${EXEC} -i:1000 -o:81920
./${EXEC} -i:1000 -o:819200
./${EXEC} -i:1000 -o:819200
./${EXEC} -i:1000 -o:8192000
./${EXEC} -i:1000 -o:8192000
./${EXEC} -i:1000 -o:16384000
./${EXEC} -i:1000 -o:16384000

./${EXEC} -i:512 -o:1024
./${EXEC} -i:512 -o:1024
./${EXEC} -i:512 -o:4096
./${EXEC} -i:512 -o:4096
./${EXEC} -i:512 -o:8192
./${EXEC} -i:512 -o:8192
./${EXEC} -i:512 -o:81920
./${EXEC} -i:512 -o:81920
./${EXEC} -i:512 -o:819200
./${EXEC} -i:512 -o:819200
./${EXEC} -i:512 -o:8192000
./${EXEC} -i:512 -o:8192000
./${EXEC} -i:512 -o:16384000
./${EXEC} -i:512 -o:16384000

