package edu.wisc.cs.sdn.vnet.rt;

import net.floodlightcontroller.packet.IPv4;
import edu.wisc.cs.sdn.vnet.Iface;

/**
 * An entry in a route table.
 * @author Aaron Gember-Jacobson and Anubhavnidhi Abhashkumar
 */
public class RouteEntry
{
	/** Destination IP address */
	private int destinationAddress;

	/** Gateway IP address (0 for directly connected subnets) */
	private int gatewayAddress;

	/** Subnet mask */
	private int maskAddress;

	/** Router interface out which packets should be sent */
	private Iface iface;

	/** RIP metric (hop count); 0 for static routes */
	private int metric;

	/** Timestamp of last RIP update (ms since epoch); used for timeouts */
	private long timestamp;

	/**
	 * Create a route table entry (static, no metric).
	 */
	public RouteEntry(int destinationAddress, int gatewayAddress,
			int maskAddress, Iface iface)
	{
		this(destinationAddress, gatewayAddress, maskAddress, iface, 0);
	}

	/**
	 * Create a route table entry with a RIP metric.
	 */
	public RouteEntry(int destinationAddress, int gatewayAddress,
			int maskAddress, Iface iface, int metric)
	{
		this.destinationAddress = destinationAddress;
		this.gatewayAddress = gatewayAddress;
		this.maskAddress = maskAddress;
		this.iface = iface;
		this.metric = metric;
		this.timestamp = System.currentTimeMillis();
	}

	public int getDestinationAddress()
	{ return this.destinationAddress; }

	public int getGatewayAddress()
	{ return this.gatewayAddress; }

	public void setGatewayAddress(int gatewayAddress)
	{ this.gatewayAddress = gatewayAddress; }

	public int getMaskAddress()
	{ return this.maskAddress; }

	public Iface getInterface()
	{ return this.iface; }

	public void setInterface(Iface iface)
	{ this.iface = iface; }

	public int getMetric()
	{ return this.metric; }

	public void setMetric(int metric)
	{ this.metric = metric; }

	public long getTimestamp()
	{ return this.timestamp; }

	/** Refresh the timestamp to now (used when a RIP update is received). */
	public void updateTimestamp()
	{ this.timestamp = System.currentTimeMillis(); }

	public String toString()
	{
		return String.format("%s \t%s \t%s \t%s",
				IPv4.fromIPv4Address(this.destinationAddress),
				IPv4.fromIPv4Address(this.gatewayAddress),
				IPv4.fromIPv4Address(this.maskAddress),
				this.iface.getName());
	}
}
