open ManagedMultiply
open System
open MathNet.Numerics.LinearAlgebra

[<EntryPoint>]
let main argv =
    let N = 1024

    // Compute the same multiplication with all different algorithms and make sure the results are equal.
    let A = DenseVector.randomStandard (N*N) |> Vector.toArray
    let B = DenseVector.randomStandard (N*N) |> Vector.toArray
    
    // Divide and conquer async parallel -> Really slow...
    (*
    let C5 = Array.zeroCreate (N*N)
    let T5 = 64
    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    DivideConquerParallel.divideAndConquer A B C5 N T5
    stopWatch.Stop()
    printfn "Divide Conquer Parallel Async: %f" stopWatch.Elapsed.TotalMilliseconds
    *)

    // Divide and conquer task parallel
    let C4 = Array.zeroCreate (N*N)
    let T4 = 64
    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    DivideConquerParallelTask.divideAndConquer A B C4 N T4
    stopWatch.Stop()
    printfn "Divide Conquer Parallel Task: %f" stopWatch.Elapsed.TotalMilliseconds

    // Regular naive three loops matrix multiply
    let C1 = Array.zeroCreate (N*N)
    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    ThreeLoops.mult A B C1 N
    stopWatch.Stop()
    printfn "ThreeLoops: %f" stopWatch.Elapsed.TotalMilliseconds

    // Tiled multiplication, implicit blocking.
    let C2 = Array.zeroCreate (N*N)
    let T2 = 64
    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    ThreeLoops.tiledMult A B C2 N T2
    stopWatch.Stop()
    printfn "Tiled: %f" stopWatch.Elapsed.TotalMilliseconds

    // Divide and conquer single threaded
    let C3 = Array.zeroCreate (N*N)
    let T5 = 64
    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    DivideConquer.divideAndConquer A B C3 N T5
    stopWatch.Stop()
    printfn "Divide Conquer Single: %f" stopWatch.Elapsed.TotalMilliseconds

    // If sequences are equal, then the results below are all 0.    
    let compareSequences = Array.forall2 (fun (elem1: float) (elem2: float) -> Math.Abs(elem1 - elem2) < 1e-8)
    let eq2 = compareSequences C1 C2
    let eq3 = compareSequences C1 C3
    let eq4 = compareSequences C1 C4
    //let eq5 = compareSequences C1 C5
    Console.WriteLine(eq2)
    Console.WriteLine(eq3)
    Console.WriteLine(eq4)
    //Console.WriteLine(eq5)


    0

