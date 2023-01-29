import socket
import threading

def port_scanner(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((host, port))
    if result == 0:
        print(f'[+] Port {port} is open')
    else:
        print(f'[-] Port {port} is closed or filtered')
    sock.close()

def main():
    host = input('Enter the host to scan: ')
    start_port = int(input('Enter the starting port: '))
    end_port = int(input('Enter the ending port: '))
    for port in range(start_port, end_port):
        t = threading.Thread(target=port_scanner, args=(host, port))
        t.start()

if __name__ == '__main__':
    main()
