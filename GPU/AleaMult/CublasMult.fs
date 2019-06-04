module CublasMult

open MathNet.Numerics.LinearAlgebra
open BenchmarkDotNet.Attributes
open MathNet.Numerics.Providers
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra.Double

type Benchmark() =

    let mutable A: float[] = null
    let mutable B: float[] = null
    let mutable C: float[] = null

    [<Params(4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)>]
    member val N = 0 with get, set

    [<GlobalSetup>]
    member this.Setup() =
        A <- DenseVector.randomStandard (this.N*this.N) |> Vector.toArray
        B <- DenseVector.randomStandard (this.N*this.N) |> Vector.toArray

    [<Benchmark>]
    member this.Benchmark() =
        C <- CudaMult.gemm A B this.N
        C
