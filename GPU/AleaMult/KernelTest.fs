module KernelTest

open Alea
open Alea.FSharp

let arrayPlusTest =

    let length = 10000

    let harg1 = Array.init length id
    let harg2 = Array.init length id
    let expected = (harg1, harg2) ||> Array.map2 (+)

    let gpu = Gpu.Default
    let darg1 = gpu.Allocate(harg1)
    let darg2 = gpu.Allocate(harg2)
    let dresult = gpu.Allocate<int>(length)

    let lp = LaunchParam(16, 256)
    gpu.Launch BlockedMult.exampleKernel lp dresult darg1 darg2
    let actual = Gpu.CopyToHost(dresult)

    Gpu.Free(darg1)
    Gpu.Free(darg2)
    Gpu.Free(dresult)

let nopTest sz =
    let N = sz*sz

    let (harg1: float32[]) = Array.init N (fun i -> float32(i))

    let gpu = Gpu.Default
    let darg1 = gpu.Allocate(harg1)
    let dresult = gpu.Allocate<float32>(N)

    let nopKernel = 
        <@ fun (arg:float32[]) (dresult:float32[]) ->
            let nCols = gridDim.y * blockDim.y
            let xcor = blockIdx.x * blockDim.x + threadIdx.x // The column
            let ycor = blockIdx.y * blockDim.y + threadIdx.y // The row

            dresult.[ycor*nCols + xcor] <- arg.[ycor*nCols + xcor] // Row-major.
        @>
        |> Compiler.makeKernel

    // Example calc. 512*512 = 262144 coordinates.
    // Need to reach this number of threads.

    // One block processes 8*8=64 threads.
    let dimThreads = 8 // 8*8 = 64 = 2 warps of 32
    let threadsPerBlock = dim3(dimThreads, dimThreads)

    // Grid dimension, total number of blocks = 64*64 = 4096.
    // Total number of threads = 4096*64 = 262144 = number of coordinates.
    let dimBlocks = sz / dimThreads
    let numBlocks = dim3(dimBlocks, dimBlocks)

    let lp = LaunchParam(numBlocks, threadsPerBlock)
    gpu.Launch nopKernel lp darg1 dresult
    let actual = Gpu.CopyToHost(dresult)

    Gpu.Free(darg1)
    Gpu.Free(dresult)


