package edu.wisc.cs.sdn.vnet;

import edu.wisc.cs.sdn.vnet.rt.Router;
import edu.wisc.cs.sdn.vnet.sw.Switch;
import edu.wisc.cs.sdn.vnet.vns.VNSComm;

/**
 * Main entry point for virtual network client.
 * @author Aaron Gember-Jacobson
 */
public class Main
{
	private static void usage()
	{
		System.out.println("Virtual Network Client");
		System.out.println("Usage: java -jar VirtualNetwork.jar -v host [-s server] [-p port] [-r routing_table] [-a arp_cache] [-l log_file]");
		System.out.println("Options:");
		System.out.println("  -v host         Device name (must start with 's' or 'r')");
		System.out.println("  -s server       Server hostname (default: localhost)");
		System.out.println("  -p port         Port number (default: 8888)");
		System.out.println("  -r routing_table  Route table file for routers");
		System.out.println("  -a arp_cache    ARP cache file for routers");
		System.out.println("  -l log_file     PCAP dump file");
	}

	public static void main(String[] args)
	{
		String host = null;
		String server = "localhost";
		String rtable = null;
		String arpcache = null;
		String logfile = null;
		int port = 8888;

		// Parse command-line arguments
		for (int i = 0; i < args.length; i++)
		{
			if (args[i].equals("-v") && i + 1 < args.length)
			{ host = args[++i]; }
			else if (args[i].equals("-s") && i + 1 < args.length)
			{ server = args[++i]; }
			else if (args[i].equals("-p") && i + 1 < args.length)
			{ port = Integer.parseInt(args[++i]); }
			else if (args[i].equals("-r") && i + 1 < args.length)
			{ rtable = args[++i]; }
			else if (args[i].equals("-a") && i + 1 < args.length)
			{ arpcache = args[++i]; }
			else if (args[i].equals("-l") && i + 1 < args.length)
			{ logfile = args[++i]; }
			else if (args[i].equals("-h"))
			{ usage(); System.exit(0); }
		}

		if (null == host)
		{
			System.err.println("Must specify device name with -v");
			usage();
			System.exit(1);
		}

		// Open PCAP log file if specified
		DumpFile dump = null;
		if (logfile != null)
		{
			dump = DumpFile.open(logfile);
			if (null == dump)
			{ System.exit(1); }
		}

		// Create device
		Device device;
		if (host.startsWith("s"))
		{
			device = new Switch(host, dump);
		}
		else if (host.startsWith("r"))
		{
			Router router = new Router(host, dump);
			device = router;

			// Connect to VNS server first so interfaces are initialized
			VNSComm vnsComm = new VNSComm(device);
			if (!vnsComm.connectToServer(port, server))
			{ System.exit(1); }

			// Now configure routing: static table or RIP
			if (rtable != null)
			{
				router.loadRouteTable(rtable);
				if (arpcache != null)
				{ router.loadArpCache(arpcache); }
			}
			else
			{
				// Start RIP (interfaces are available after connectToServer)
				router.startRip();
			}

			vnsComm.readFromServer();
			device.destroy();
			return;
		}
		else
		{
			System.err.println("Device name must start with 's' or 'r'");
			System.exit(1);
			return;
		}

		// For switches
		VNSComm vnsComm = new VNSComm(device);
		if (!vnsComm.connectToServer(port, server))
		{ System.exit(1); }
		vnsComm.readFromServer();
		device.destroy();
	}
}
