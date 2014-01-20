# -*- coding: utf-8 -*-

# Sampling datasets from RawData
langs = {0: "sp", 1: "en"}
lang = langs[1]

# Load corpus from domains review
with open("%s_sampling.txt"%lang, "w") as file_out:
    # Load corpus from hotels review
    cont = 0
    with open("data/rawdata/%s/electronics/electronics.txt" % lang, "r") as file_in:
        for line in file_in.readlines():
            if cont % 3 == 0:
                file_out.write(line)
            cont += 1



