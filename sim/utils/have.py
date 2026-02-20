import dpkt
with open('evidence.pcapng', 'rb') as file:
    pcap = dpkt.pcapng.Reader(file)
    for ts, buf in pcap:
        eth = dpkt.ethernet.Ethernet(buf)
        ip = eth.data
        tcp = ip.data
        print(ip)
        # print(tcp)
    file.close()