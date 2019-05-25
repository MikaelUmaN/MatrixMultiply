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


