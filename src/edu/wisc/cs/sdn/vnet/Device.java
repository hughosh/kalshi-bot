package edu.wisc.cs.sdn.vnet;

import java.util.HashMap;
import java.util.Map;

import net.floodlightcontroller.packet.Ethernet;

import edu.wisc.cs.sdn.vnet.vns.VNSComm;

/**
 * @author Aaron Gember-Jacobson
 */
public abstract class Device
{
	/** Hostname for the device */
	private String host;

	/** List of the device's interfaces; maps interface name's to interfaces */
	protected Map<String,Iface> interfaces;

	/** PCAP dump file for logging all packets sent/received by the device;
	 *  null if packets should not be logged */
	private DumpFile logfile;

	/** Virtual Network Simulator communication manager for the device */
	private VNSComm vnsComm;

	/**
	 * Creates a device.
	 * @param host hostname for the device
	 * @param logfile PCAP dump file for logging all packets sent/received by the device
	 */
	public Device(String host, DumpFile logfile)
	{
		this.host = host;
		this.logfile = logfile;
		this.interfaces = new HashMap<String,Iface>();
		this.vnsComm = null;
	}

	public void setLogFile(DumpFile logfile)
	{ this.logfile = logfile; }

	public DumpFile getLogFile()
	{ return this.logfile; }

	public String getHost()
	{ return this.host; }

	public Map<String,Iface> getInterfaces()
	{ return this.interfaces; }

	public void setVNSComm(VNSComm vnsComm)
	{ this.vnsComm = vnsComm; }

	public void destroy()
	{
		if (logfile != null)
		{ this.logfile.close(); }
	}

	public Iface addInterface(String ifaceName)
	{
		Iface iface = new Iface(ifaceName);
		this.interfaces.put(ifaceName, iface);
		return iface;
	}

	public Iface getInterface(String ifaceName)
	{ return this.interfaces.get(ifaceName); }

	public boolean sendPacket(Ethernet etherPacket, Iface iface)
	{ return this.vnsComm.sendPacket(etherPacket, iface.getName()); }

	public abstract void handlePacket(Ethernet etherPacket, Iface inIface);
}
