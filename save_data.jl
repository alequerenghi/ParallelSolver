using BenchmarkTools
using Statistics
using DataFrames
using CSV
using MPI

function savedata(csv_file, t::Vector,  n, testtype, comm=nothing, overlap=0)
    df = DataFrame(
        n_nodes=n^2,
        comm_size=isnothing(comm) ? 1 : MPI.Comm_size(comm),
        min_iter = minimum(t),
        mean_iter = mean(t),
        max_iter = maximum(t),
        cpu_name=Sys.cpu_info()[1].model,
        problem_name=testtype,
        overlap=overlap,
    )

    CSV.write(csv_file, df; append=isfile(csv_file))
end

function savedata(csv_file, t::BenchmarkTools.Trial, n, testtype, comm::MPI.Comm, overlap)
    mem_tot = MPI.Reduce(t.memory, MPI.SUM, comm)
    allocs_tot = MPI.Reduce(t.allocs, MPI.SUM, comm)

    if 0 == MPI.Comm_rank(comm)
        t.memory = mem_tot
        t.allocs = allocs_tot

        savedata(csv_file, t, n, testtype, MPI.Comm_size(comm), overlap)
    end
end

function savedata(csv_file, t::BenchmarkTools.Trial, n, testtype, commsize=1, overlap=0)
    mn = minimum(t)
    med = median(t)
    avg = mean(t)
    mx = maximum(t)

    df = DataFrame(
        n_nodes=n^2,
        comm_size=commsize,
        min_time_s=mn.time/1e9,
        med_time_s=med.time/1e9,
        mean_time_s=avg.time/1e9,
        max_time_s=mx.time/1e9,
        memory_mb=med.memory/2^20,
        allocations=med.allocs,
        cpu_name=Sys.cpu_info()[1].model,
        problem_name=testtype,
        overlap=overlap,
    )

    CSV.write(csv_file, df; append=isfile(csv_file))
end

function savedata(
        n::Int,
        commsize::Int,
        min_time_s,
        median_time_s,
        mean_time_s,
        max_time_s,
        total_mem_mb,
        total_allocs,
        csv_file,
        testtype,
)
        cpuname  = Sys.cpu_info()[1].model

        df = DataFrame(
            n_nodes = n,
            comm_size = commsize,
            min_time_sec = min_time_s,
            median_time_sec = median_time_s,
            mean_time_sec = mean_time_s,
            max_time_sec = max_time_s,
            total_memory_mb = total_mem_mb,
            allocations = total_allocs,
            cpu_name = cpuname,
            problem_name = testtype
        )

        CSV.write(csv_file, df; append=isfile(csv_file))
    return nothing
end
