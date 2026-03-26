package edu.wisc.cs.sdn.vnet.rt;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import net.floodlightcontroller.packet.IPv4;

import edu.wisc.cs.sdn.vnet.Iface;

/**
 * Route table for a router.
 * @author Aaron Gember-Jacobson
 */
public class RouteTable
{
	/** Entries in the route table */
	private List<RouteEntry> entries;

	public RouteTable()
	{ this.entries = new ArrayList<>(); }

	/**
	 * Lookup the route entry with the longest prefix match for a given IP.
	 * @param ip destination IP address
	 * @return best matching route entry, or null if none
	 */
	public RouteEntry lookup(int ip)
	{
		synchronized (this.entries)
		{
			RouteEntry best = null;
			int bestMask = 0;
			for (RouteEntry entry : this.entries)
			{
				int mask = entry.getMaskAddress();
				if ((ip & mask) == entry.getDestinationAddress())
				{
					// Pick the entry with the longest (most specific) mask
					if (best == null || Integer.compareUnsigned(mask, bestMask) > 0)
					{
						best = entry;
						bestMask = mask;
					}
				}
			}
			return best;
		}
	}

	/**
	 * Find an exact entry by destination IP and mask.
	 * @return matching entry, or null if none
	 */
	public RouteEntry find(int dstIp, int maskIp)
	{
		synchronized (this.entries)
		{
			for (RouteEntry entry : this.entries)
			{
				if (entry.getDestinationAddress() == dstIp
						&& entry.getMaskAddress() == maskIp)
				{ return entry; }
			}
			return null;
		}
	}

	/**
	 * Get a snapshot of all route entries (for RIP advertisement).
	 */
	public List<RouteEntry> getEntries()
	{
		synchronized (this.entries)
		{ return new ArrayList<>(this.entries); }
	}

	/**
	 * Load the route table from a file.
	 * Format: destination gateway mask interface
	 */
	public boolean load(String filename, Router router)
	{
		BufferedReader reader;
		try
		{
			reader = new BufferedReader(new FileReader(filename));
		}
		catch (FileNotFoundException e)
		{
			System.err.println(e.toString());
			return false;
		}

		while (true)
		{
			String line;
			try { line = reader.readLine(); }
			catch (IOException e)
			{
				System.err.println(e.toString());
				try { reader.close(); } catch (IOException f) {}
				return false;
			}
			if (null == line) { break; }

			String ipPat = "(\\d+\\.\\d+\\.\\d+\\.\\d+)";
			String ifPat = "([a-zA-Z0-9]+)";
			Pattern pattern = Pattern.compile(
					String.format("%s\\s+%s\\s+%s\\s+%s",
							ipPat, ipPat, ipPat, ifPat));
			Matcher matcher = pattern.matcher(line);
			if (!matcher.matches() || matcher.groupCount() != 4)
			{
				System.err.println("Invalid entry in routing table file");
				try { reader.close(); } catch (IOException f) {}
				return false;
			}

			int dstIp = IPv4.toIPv4Address(matcher.group(1));
			if (0 == dstIp)
			{
				System.err.println("Error loading route table, cannot convert "
						+ matcher.group(1) + " to valid IP");
				try { reader.close(); } catch (IOException f) {}
				return false;
			}

			int gwIp = IPv4.toIPv4Address(matcher.group(2));

			int maskIp = IPv4.toIPv4Address(matcher.group(3));
			if (0 == maskIp)
			{
				System.err.println("Error loading route table, cannot convert "
						+ matcher.group(3) + " to valid IP");
				try { reader.close(); } catch (IOException f) {}
				return false;
			}

			String ifaceName = matcher.group(4).trim();
			Iface iface = router.getInterface(ifaceName);
			if (null == iface)
			{
				System.err.println("Error loading route table, invalid interface "
						+ matcher.group(4));
				try { reader.close(); } catch (IOException f) {}
				return false;
			}

			this.insert(dstIp, gwIp, maskIp, iface);
		}

		try { reader.close(); } catch (IOException f) {}
		return true;
	}

	/**
	 * Add a static route entry (no metric).
	 */
	public void insert(int dstIp, int gwIp, int maskIp, Iface iface)
	{
		insert(dstIp, gwIp, maskIp, iface, 0);
	}

	/**
	 * Add a route entry with a RIP metric.
	 */
	public void insert(int dstIp, int gwIp, int maskIp, Iface iface, int metric)
	{
		RouteEntry entry = new RouteEntry(dstIp, gwIp, maskIp, iface, metric);
		synchronized (this.entries)
		{ this.entries.add(entry); }
	}

	/**
	 * Remove an entry by destination IP and mask.
	 * @return true if an entry was removed
	 */
	public boolean remove(int dstIp, int maskIp)
	{
		synchronized (this.entries)
		{
			RouteEntry entry = this.find(dstIp, maskIp);
			if (null == entry) { return false; }
			this.entries.remove(entry);
			return true;
		}
	}

	/**
	 * Update gateway and interface for an existing entry.
	 */
	public boolean update(int dstIp, int maskIp, int gwIp, Iface iface)
	{
		synchronized (this.entries)
		{
			RouteEntry entry = this.find(dstIp, maskIp);
			if (null == entry) { return false; }
			entry.setGatewayAddress(gwIp);
			entry.setInterface(iface);
			return true;
		}
	}

	public String toString()
	{
		synchronized (this.entries)
		{
			if (this.entries.isEmpty())
			{ return " WARNING: route table empty"; }
			StringBuilder sb = new StringBuilder(
					"Destination\tGateway\t\tMask\t\tIface\n");
			for (RouteEntry entry : this.entries)
			{ sb.append(entry.toString()).append("\n"); }
			return sb.toString();
		}
	}
}
