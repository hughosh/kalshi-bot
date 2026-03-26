package edu.wisc.cs.sdn.vnet.sw;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.MACAddress;

import edu.wisc.cs.sdn.vnet.Device;
import edu.wisc.cs.sdn.vnet.DumpFile;
import edu.wisc.cs.sdn.vnet.Iface;

/**
 * Virtual switch with MAC learning and 15-second table timeout.
 * @author Aaron Gember-Jacobson
 */
public class Switch extends Device implements Runnable
{
	/** Forwarding table: MAC -> interface */
	private ConcurrentHashMap<MACAddress, Iface> forwardingTable;

	/** Timestamps for forwarding table entries (nanoseconds) */
	private ConcurrentHashMap<MACAddress, Long> tableTimestamps;

	/** Background thread that flushes stale entries */
	private Thread flushThread;

	/** Entry timeout: 15 seconds in nanoseconds */
	private static final long TIMEOUT_NS = 15_000_000_000L;

	public Switch(String host, DumpFile logfile)
	{
		super(host, logfile);
		this.forwardingTable = new ConcurrentHashMap<>();
		this.tableTimestamps = new ConcurrentHashMap<>();
		this.flushThread = new Thread(this, "SwitchTableFlush");
		this.flushThread.setDaemon(true);
		this.flushThread.start();
	}

	/** Background task: remove entries older than TIMEOUT_NS. */
	@Override
	public void run()
	{
		try
		{
			while (true)
			{
				Thread.sleep(1000);
				long now = System.nanoTime();
				for (Map.Entry<MACAddress, Long> entry : tableTimestamps.entrySet())
				{
					if (now - entry.getValue() > TIMEOUT_NS)
					{
						MACAddress mac = entry.getKey();
						tableTimestamps.remove(mac);
						forwardingTable.remove(mac);
					}
				}
			}
		}
		catch (InterruptedException e)
		{
			System.out.println("Switch flush thread interrupted: " + e);
		}
	}

	/**
	 * Handle an Ethernet packet received on a specific interface.
	 * @param etherPacket the Ethernet packet that was received
	 * @param inIface the interface on which the packet was received
	 */
	@Override
	public void handlePacket(Ethernet etherPacket, Iface inIface)
	{
		System.out.println("*** -> Received packet: " +
				etherPacket.toString().replace("\n", "\n\t"));

		MACAddress srcMac = etherPacket.getSourceMAC();
		MACAddress dstMac = etherPacket.getDestinationMAC();
		long now = System.nanoTime();

		// Learn source MAC -> interface mapping
		forwardingTable.put(srcMac, inIface);
		tableTimestamps.put(srcMac, now);

		// Lookup destination
		Iface outIface = null;
		if (tableTimestamps.containsKey(dstMac))
		{
			outIface = forwardingTable.get(dstMac);
		}

		if (outIface == null)
		{
			// Flood out all interfaces except the one it arrived on
			for (Iface iface : this.interfaces.values())
			{
				if (iface != inIface)
				{ sendPacket(etherPacket, iface); }
			}
		}
		else
		{
			// Forward to known interface
			sendPacket(etherPacket, outIface);
		}
	}
}
