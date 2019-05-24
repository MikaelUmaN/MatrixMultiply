module TiledBenchmark

open MathNet.Numerics.LinearAlgebra
open BenchmarkDotNet.Attributes
open ManagedMultiply

[<RPlotExporter; RankColumn>]
type Benchmark() =

    let mutable A: float[] = null
    let mutable B: float[] = null
    let mutable C: float[] = null

    [<Params(4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)>]
    member val N = 0 with get, set

    [<Params(16, 32, 64, 128, 256)>]
    member val T = 0 with get, set

    [<GlobalSetup>]
    member this.Setup() =
        A <- DenseVector.randomStandard (this.N*this.N) |> Vector.toArray
        B <- DenseVector.randomStandard (this.N*this.N) |> Vector.toArray
        C <- Array.zeroCreate (this.N*this.N)

    [<Benchmark>]
    member this.Benchmark() =
        ThreeLoops.tiledMult A B C this.N this.T
        C
