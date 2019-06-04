module BlockedMult

open Alea
open Alea.FSharp

let exampleKernel = 
    <@ fun (result:int[]) (arg1:int[]) (arg2:int[]) ->
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start
        while i < result.Length do
            result.[i] <- arg1.[i] + arg2.[i]
            i <- i + stride @>
    |> Compiler.makeKernel


let mult (A:float[]) (B:float[]) (N:int) =
    let gpu = Gpu.Default
    let Ag = gpu.Allocate(A)
    let Bg = gpu.Allocate(B)
    let Cg = gpu.Allocate<float>(N*N)

    // One thread computes one coordinate of C in full.
    let naiveMatrixMult = 
        <@ fun (Ag:float[]) (Bg:float[]) (Cg:float[]) ->
            //let nCols = gridDim.y * blockDim.y
            let xcor = blockIdx.x * blockDim.x + threadIdx.x // The row
            let ycor = blockIdx.y * blockDim.y + threadIdx.y // The column

            if (xcor >= N || ycor >= N) then
                () // Outside the matrix boundaries, do nothing.
            else
                let mutable s = 0.
                for k in 0..N-1 do
                    s <- s + Ag.[xcor*N + k] * Bg.[k*N + ycor] // Row-major.
                Cg.[xcor*N + ycor] <- s
        @>
        |> Compiler.makeKernel

    // One block processes 8*8=64 threads.
    let dimThreads = 8 // 8*8 = 64 = 2 warps of 32
    let threadsPerBlock = dim3(dimThreads, dimThreads)

    // Grid dimension. Matrices are square, so we only take one dimension into account.
    // 4 threads per blocks -> we need N/threads blocks.
    // E.g. for 1024 we need 256 blocks.
    let dimBlocks = N / dimThreads
    let numBlocks = dim3(dimBlocks, dimBlocks)

    let lp = LaunchParam(numBlocks, threadsPerBlock)
    gpu.Launch naiveMatrixMult lp Ag Bg Cg
    let actual = Gpu.CopyToHost(Cg)

    Gpu.Free(Ag)
    Gpu.Free(Bg)
    Gpu.Free(Cg)

    // Calculated C
    actual

