package edu.wisc.cs.sdn.vnet.rt;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.MACAddress;

/**
 * ARP cache for a router.
 * @author Aaron Gember-Jacobson
 */
public class ArpCache
{
	/** Cached IP-to-MAC mappings */
	private ConcurrentHashMap<Integer, ArpEntry> entries;

	public ArpCache()
	{ this.entries = new ConcurrentHashMap<>(); }

	/**
	 * Add an IP-to-MAC mapping to the cache.
	 */
	public void insert(MACAddress mac, int ip)
	{ this.entries.put(ip, new ArpEntry(mac, ip)); }

	/**
	 * Look up a MAC address for a given IP.
	 * @return the ArpEntry, or null if not found
	 */
	public ArpEntry lookup(int ip)
	{ return this.entries.get(ip); }

	/**
	 * Load the ARP cache from a file.
	 * Each line: IP_ADDRESS MAC_ADDRESS
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
			line = line.trim();
			if (line.isEmpty()) { continue; }

			String ipPattern = "(\\d+\\.\\d+\\.\\d+\\.\\d+)";
			String macPattern = "([0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}" +
					":[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2})";
			Pattern pattern = Pattern.compile(
					String.format("%s\\s+%s", ipPattern, macPattern));
			Matcher matcher = pattern.matcher(line);
			if (!matcher.matches() || matcher.groupCount() != 2)
			{
				System.err.println("Invalid entry in ARP cache file: " + line);
				continue;
			}

			int ip = IPv4.toIPv4Address(matcher.group(1));
			if (0 == ip)
			{
				System.err.println("Invalid IP in ARP cache: " + matcher.group(1));
				continue;
			}

			MACAddress mac = MACAddress.valueOf(matcher.group(2));
			insert(mac, ip);
		}

		try { reader.close(); } catch (IOException e) {}
		return true;
	}

	public String toString()
	{
		if (this.entries.isEmpty())
		{ return " WARNING: ARP cache empty"; }

		StringBuilder sb = new StringBuilder("IP\t\t\tMAC\n");
		for (ArpEntry entry : this.entries.values())
		{ sb.append(entry.toString()).append("\n"); }
		return sb.toString();
	}
}
