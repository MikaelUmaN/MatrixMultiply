module CudaMult

open Alea
open Alea.FSharp
open Alea.cuBLAS

// y <- ax + y
let axpy (A:float32[]) (B:float32[]) (N:int) =
    let gpu = Gpu.Default
    let Ag = gpu.AllocateDevice(A)
    let Bg = gpu.AllocateDevice(B)
    //let Cg = gpu.AllocateDevice<float32>(N*N)

    let blas = new Blas(gpu)
    
    let alpha = 1.f // Scale factor in axpy: y <- a*x + y
    blas.Axpy(N, alpha, Ag.Ptr, 1, Bg.Ptr, 1)
    
    //blas.Gemm(Operation.N, Operation.N, N, N, N, alpha, Ag, 1, Bg, 1, 0, Cg, 1)
 
    // Calculated C
    let actual = Gpu.CopyToHost(Bg)
    actual

// General matrix multiplication
// C <- a*op(A)op(B) + beta*C
let gemm (A:float[]) (B:float[]) (N:int) =
    let gpu = Gpu.Default
    let Ag = gpu.AllocateDevice(A)
    let Bg = gpu.AllocateDevice(B)
    let Cg = gpu.AllocateDevice<float>(N*N)

    let blas = new Blas(gpu)
    
    // Scale factors.
    let alpha = 1.
    let beta = 1.
    
    blas.Gemm(Operation.N, Operation.N, N, N, N, alpha, Ag.Ptr, N, Bg.Ptr, N, beta, Cg.Ptr, N)
 
    // Calculated C
    let actual = Gpu.CopyToHost(Cg)
    actual
