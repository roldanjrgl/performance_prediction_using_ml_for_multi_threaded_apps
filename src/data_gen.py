import subprocess
import os


def run_parsec():
    parsec = {
        "benchmarks": ["blackscholes", "bodytrack", "canneal", "facesim", "ferret", "fluidanimate",
                       "streamcluster", "swaptions", "vips", "x264"],
        "input_sizes": ["simsmall", "simmedium", "simlarge"],
        "threads": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 64, 128]
    }
    for benchmark in parsec["benchmarks"]:
        run_parsec_benchmark(benchmark, parsec)


def run_parsec_benchmark(benchmark, specs):
    single_thread_time = 0
    for input_size in specs["input_sizes"]:
        for thread in specs["threads"]:
            print("Running: " + benchmark + " thread: " + str(thread) + " input size:" + str(input_size) + "\n")
            file_stream = open("data.csv", "a")
            output_string = []
            print("parsecmgmt -a run -p " + benchmark + " -n " + str(thread) + " -i " + input_size + "\n")
            cmd = subprocess.Popen("parsecmgmt -a run -p " + benchmark + " -n " + str(thread) + " -i " + input_size,
                                   shell=True, stdout=subprocess.PIPE)
            cmd.wait()
            real_time_taken = real_time_parser(cmd.stdout.readline)

            if thread == 1:
                single_thread_time = real_time_taken
            speed_up = single_thread_time / real_time_taken
            output_string.append("parsec")
            output_string.append(benchmark)
            output_string.append(str(thread))
            output_string.append(input_size)
            print( "perf stat -o perf.txt --field-separator=, -e branch-instructions,branch-misses,cache-misses,"
                "cache-references,cycles,instructions,cpu-clock,page-faults,L1-dcache-loads,L1-icache-load-misses,"
                "LLC-load-misses -- parsecmgmt -a run -p "
                  + benchmark + " -n " + str(thread) + "-i " + input_size)
            cmd = subprocess.Popen(
               "perf stat -o perf.txt --field-separator=, -e branch-instructions,branch-misses,cache-misses,"
                "cache-references,cycles,instructions,cpu-clock,page-faults,L1-dcache-loads,L1-icache-load-misses,"
                "LLC-load-misses --  parsecmgmt -a run -p " + benchmark + " -n " + str(
                    thread) + " -i " + input_size, shell=True, stdout=subprocess.PIPE)
            cmd.wait()
            parse_perf(output_string, real_time_taken, speed_up, file_stream)
            file_stream.close()


def parse_perf(output_string, real_time_taken, speed_up, file_stream):
    with open("perf.txt", "r") as perf_text:
        perf = perf_text.readlines()
    for line in perf[2:]:
        line_values = line.split(",")
        metric_name = line_values[2]
        metric_value = line_values[0]
        if "instructions" in metric_name or "cache-misses" in metric_name or "branch-misses" in metric_name:
            output_string.append(metric_value)
            output_string.append(line_values[5])
        else:
            output_string.append(metric_value)
    output_string.append(str(real_time_taken))
    output_string.append(str(speed_up))
    file_stream.write(",".join(output_string) + "\n")


def run_splash():
    splash = {
        "benchmarks": ["BARNES", "FMM", "OCEAN", "RADIOSITY", "VOLREND", "WATER-NSQUARED", "WATER-SPATIAL"]
    }
    for benchmark in splash["benchmarks"]:
        run_splash_benchmark(benchmark)


def run_splash_benchmark(benchmark):
    for (input_path, dir_name, file_name) in os.walk(benchmark.lower() + "/inputs"):
        file_stream = open("data.csv", "a")
        cmd_string = ""
        thread_count = 1
        single_thread_time = 0
        if benchmark == "BARNES":
            cmd_string = "./" + benchmark.lower() + "/inputs/BARNES < " + "./" + benchmark.lower() \
                         + "/inputs/" + file_name
            thread_count = file_name.split("-p")[1]
        elif benchmark == "FMM":
            cmd_string = "./" + benchmark.lower() + "/inputs/FMM < " + "./" + benchmark.lower() \
                         + "/inputs/" + file_name
            thread_count = file_name.split(".")[1]
        elif benchmark == "WATER-NSQUARED":
            cmd_string = "./" + benchmark.lower() + "/inputs/WATER-NSQUARED < " + benchmark.lower() \
                         + "/inputs/" + file_name
            thread_count = file_name.split("-p")[1]
        elif benchmark == "WATER-SPATIAL":
            cmd_string = "./" + benchmark.lower() + "/inputs/WATER-SPATIAL < " + benchmark.lower() \
                         + "/inputs/" + file_name
            thread_count = file_name.split("-p")[1]
        print("time " + cmd_string + "/n")

        cmd = subprocess.Popen("time " + cmd_string, shell=True, stdout=subprocess.PIPE)
        cmd.wait()
        real_time_taken = real_time_parser(cmd.stdout.readline)
        if thread_count == 1:
            single_thread_time = real_time_taken

        speed_up = single_thread_time / real_time_taken
        output_string = ["splash", benchmark, str(thread_count), file_name]
        print( "perf stat -o perf.txt --field-separator=, -e branch-instructions,branch-misses,cache-misses,"
                "cache-references,cycles,instructions,cpu-clock,page-faults,L1-dcache-loads,L1-icache-load-misses,"
                "LLC-load-misses -- " + cmd_string)
        cmd = subprocess.Popen(
            "perf stat -o perf.txt --field-separator=, -e branch-instructions,branch-misses,cache-misses,"
                "cache-references,cycles,instructions,cpu-clock,page-faults,L1-dcache-loads,L1-icache-load-misses,"
                "LLC-load-misses -- " + cmd_string, shell=True, stdout=subprocess.PIPE)
        cmd.wait()
        parse_perf(output_string, real_time_taken, speed_up, file_stream)
        file_stream.close()


def real_time_parser(command_line_string):
    for line in iter(command_line_string, ""):
        if b"real" in line:
            split_line = line.split(b"\t")
            real_time = split_line[1]
            m, s = real_time.split(b"m")
            s, seconds = s.split(b"s")
            real_time = int(m) * 60 + float(s)
            return real_time


def main():
    file_stream = open("data.csv", "a")
    file_stream.write("package, benchmark, threads, input_size, branch_instructions, branch_instructions_rate,"
                      "branch_misses, branch_miss_percentage, l3_cache_misses,l3_cache_miss_percentage, "
                      "l3_cache_references, cpu_cycles, total_instructions, ipc, cpu_clock, "
                      "page_faults,l1_data_cache_loads, l1_instruction_cache_load_misses , llc_load_misses, exe_time, "
                      "speedup\n")
    file_stream.close()
    run_parsec()
    # run_splash()


if __name__ == "__main__":
    main()
