
[<EntryPoint>]
let main argv = 

    // Just jitting and warmup
    for i in 5..7 do
        let sz = (int) <| 2.**float(i)
        KernelTest.nopTest sz


    for i in 6..12 do
        let sz = (int) <| 2.**float(i)
        let stopWatch = System.Diagnostics.Stopwatch.StartNew()
        KernelTest.nopTest sz
        stopWatch.Stop()
        printfn "Elapsed %A for size %A" stopWatch.ElapsedMilliseconds sz

    0 // return an integer exit code