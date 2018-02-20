module dummy
    using ParallelDataTransfer

    function myx(xval)
        global x = xval
        @everywhere global x
        passobj(myid(), workers(), :x, from_mod = dummy, to_mod = dummy)
        @everywhere println(dummy.x)
    end

end
