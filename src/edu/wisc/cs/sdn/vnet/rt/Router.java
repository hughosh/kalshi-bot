package edu.wisc.cs.sdn.vnet.rt;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ConcurrentHashMap;

import net.floodlightcontroller.packet.ARP;
import net.floodlightcontroller.packet.Data;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.ICMP;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.MACAddress;
import net.floodlightcontroller.packet.RIPv2;
import net.floodlightcontroller.packet.RIPv2Entry;
import net.floodlightcontroller.packet.UDP;

import edu.wisc.cs.sdn.vnet.Device;
import edu.wisc.cs.sdn.vnet.DumpFile;
import edu.wisc.cs.sdn.vnet.Iface;

/**
 * Virtual router that supports both static route tables and RIPv2 dynamic routing.
 *
 * Assignment 2 features:
 *   - IP forwarding with TTL decrement and checksum verification
 *   - ARP request/reply with queuing for packets awaiting MAC resolution
 *   - ICMP: echo reply, time exceeded, destination unreachable
 *
 * Assignment 3 additions:
 *   - RIPv2 distance-vector routing (RFC 2453 subset)
 *   - Fix: never route a packet back out the interface it arrived on
 *
 * @author Aaron Gember-Jacobson (framework), student implementation
 */
public class Router extends Device
{
	/** Route table for the router */
	private RouteTable routeTable;

	/** ARP cache for the router */
	private ArpCache arpCache;

	/** Packets waiting for ARP resolution: next-hop IP -> queue entry */
	private ConcurrentHashMap<Integer, ArpQueueEntry> arpQueues;

	// -----------------------------------------------------------------------
	// RIP constants
	// -----------------------------------------------------------------------
	private static final int    RIP_MULTICAST_IP = IPv4.toIPv4Address("224.0.0.9");
	private static final byte[] BROADCAST_MAC    = {
		(byte)0xFF,(byte)0xFF,(byte)0xFF,(byte)0xFF,(byte)0xFF,(byte)0xFF
	};
	private static final long   RIP_TIMEOUT_MS   = 30_000L;   // 30 seconds
	private static final long   RIP_INTERVAL_MS  = 10_000L;   // 10 seconds
	private static final int    RIP_INFINITY      = 16;

	// -----------------------------------------------------------------------
	// ARP constants
	// -----------------------------------------------------------------------
	private static final long ARP_RETRY_MS  = 1_000L;  // 1 second between retries
	private static final int  ARP_MAX_TRIES = 3;       // send at most 3 ARP requests

	// -----------------------------------------------------------------------
	// Inner class: ARP queue entry
	// -----------------------------------------------------------------------
	private class ArpQueueEntry
	{
		List<Ethernet> packets = new ArrayList<>();
		Iface outIface;   // interface to send packets out
		Iface inIface;    // interface the original packets arrived on
		int   nextHopIp;
		int   tries = 0;
		Timer retryTimer;
	}

	// -----------------------------------------------------------------------
	// Constructor
	// -----------------------------------------------------------------------
	public Router(String host, DumpFile logfile)
	{
		super(host, logfile);
		this.routeTable = new RouteTable();
		this.arpCache   = new ArpCache();
		this.arpQueues  = new ConcurrentHashMap<>();
	}

	public RouteTable getRouteTable()
	{ return this.routeTable; }

	// -----------------------------------------------------------------------
	// Static configuration (Assignment 2 mode)
	// -----------------------------------------------------------------------
	public void loadRouteTable(String rtFile)
	{
		if (!routeTable.load(rtFile, this))
		{
			System.err.println("Error setting up routing table from file " + rtFile);
			System.exit(1);
		}
		System.out.println("Loaded static route table");
		System.out.println("-------------------------------------------------");
		System.out.print(this.routeTable.toString());
		System.out.println("-------------------------------------------------");
	}

	public void loadArpCache(String arpCacheFile)
	{
		if (!arpCache.load(arpCacheFile, this))
		{
			System.err.println("Error setting up ARP cache from file " + arpCacheFile);
			System.exit(1);
		}
		System.out.println("Loaded static ARP cache");
		System.out.println("----------------------------------");
		System.out.print(this.arpCache.toString());
		System.out.println("----------------------------------");
	}

	// -----------------------------------------------------------------------
	// RIP initialization (Assignment 3 mode)
	// -----------------------------------------------------------------------
	/**
	 * Start RIP: populate the route table with directly connected subnets,
	 * send an initial RIP request, and schedule periodic advertisements.
	 * Must be called after interfaces are initialized.
	 */
	public void startRip()
	{
		// Add directly connected subnets (gateway = 0, metric = 1)
		for (Iface iface : this.interfaces.values())
		{
			int network = iface.getIpAddress() & iface.getSubnetMask();
			routeTable.insert(network, 0, iface.getSubnetMask(), iface, 1);
		}

		System.out.println("RIP started. Initial route table:");
		System.out.println("-------------------------------------------------");
		System.out.print(this.routeTable.toString());
		System.out.println("-------------------------------------------------");

		// Broadcast RIP request out all interfaces
		sendRipRequest();

		// Unsolicited response every RIP_INTERVAL_MS
		Timer responseTimer = new Timer("RipResponseTimer", true);
		responseTimer.scheduleAtFixedRate(new TimerTask()
		{
			@Override public void run() { sendRipResponseAll(); }
		}, RIP_INTERVAL_MS, RIP_INTERVAL_MS);

		// Route timeout check every second
		Timer timeoutTimer = new Timer("RipTimeoutTimer", true);
		timeoutTimer.scheduleAtFixedRate(new TimerTask()
		{
			@Override public void run() { checkRipTimeouts(); }
		}, 1000, 1000);
	}

	/** Remove RIP-learned entries older than RIP_TIMEOUT_MS. */
	private void checkRipTimeouts()
	{
		long now = System.currentTimeMillis();
		for (RouteEntry entry : routeTable.getEntries())
		{
			// Never remove directly connected subnets (gateway == 0)
			if (entry.getGatewayAddress() == 0) { continue; }
			if (now - entry.getTimestamp() > RIP_TIMEOUT_MS)
			{
				routeTable.remove(entry.getDestinationAddress(),
						entry.getMaskAddress());
			}
		}
	}

	// -----------------------------------------------------------------------
	// RIP packet sending
	// -----------------------------------------------------------------------
	/** Broadcast a RIP request out all interfaces. */
	private void sendRipRequest()
	{
		for (Iface iface : this.interfaces.values())
		{
			Ethernet pkt = buildRipPacket(
					RIPv2.COMMAND_REQUEST, iface,
					RIP_MULTICAST_IP, BROADCAST_MAC);
			sendPacket(pkt, iface);
		}
	}

	/** Broadcast an unsolicited RIP response out all interfaces. */
	private void sendRipResponseAll()
	{
		for (Iface iface : this.interfaces.values())
		{
			Ethernet pkt = buildRipPacket(
					RIPv2.COMMAND_RESPONSE, iface,
					RIP_MULTICAST_IP, BROADCAST_MAC);
			sendPacket(pkt, iface);
		}
	}

	/**
	 * Send a solicited RIP response back to a specific requester.
	 * @param outIface  interface to send out
	 * @param dstIp     requester's IP address
	 * @param dstMac    requester's MAC address
	 */
	private void sendRipResponseTo(Iface outIface, int dstIp, byte[] dstMac)
	{
		Ethernet pkt = buildRipPacket(
				RIPv2.COMMAND_RESPONSE, outIface, dstIp, dstMac);
		sendPacket(pkt, outIface);
	}

	/**
	 * Build a complete Ethernet/IPv4/UDP/RIPv2 packet.
	 * For COMMAND_RESPONSE, all current route table entries are included.
	 */
	private Ethernet buildRipPacket(byte command, Iface srcIface,
			int dstIp, byte[] dstMac)
	{
		// RIPv2 payload
		RIPv2 rip = new RIPv2();
		rip.setCommand(command);

		if (command == RIPv2.COMMAND_RESPONSE)
		{
			for (RouteEntry re : routeTable.getEntries())
			{
				RIPv2Entry ripEntry = new RIPv2Entry();
				ripEntry.setAddress(re.getDestinationAddress());
				ripEntry.setSubnetMask(re.getMaskAddress());
				ripEntry.setNextHopAddress(srcIface.getIpAddress());
				ripEntry.setMetric(re.getMetric());
				rip.addEntry(ripEntry);
			}
		}

		// UDP
		UDP udp = new UDP();
		udp.setSourcePort(UDP.RIP_PORT);
		udp.setDestinationPort(UDP.RIP_PORT);
		udp.setChecksum((short) 0);
		udp.setPayload(rip);

		// IPv4
		IPv4 ip = new IPv4();
		ip.setProtocol(IPv4.PROTOCOL_UDP);
		ip.setTtl((byte) 64);
		ip.setSourceAddress(srcIface.getIpAddress());
		ip.setDestinationAddress(dstIp);
		ip.setPayload(udp);

		// Ethernet
		Ethernet ether = new Ethernet();
		ether.setEtherType(Ethernet.TYPE_IPv4);
		ether.setSourceMACAddress(srcIface.getMacAddress().toBytes());
		ether.setDestinationMACAddress(dstMac);
		ether.setPayload(ip);

		return ether;
	}

	// -----------------------------------------------------------------------
	// Packet dispatch
	// -----------------------------------------------------------------------
	@Override
	public void handlePacket(Ethernet etherPacket, Iface inIface)
	{
		System.out.println("*** -> Received packet: " +
				etherPacket.toString().replace("\n", "\n\t"));

		switch (etherPacket.getEtherType())
		{
			case Ethernet.TYPE_IPv4:
				handleIpPacket(etherPacket, inIface);
				break;
			case Ethernet.TYPE_ARP:
				handleArpPacket(etherPacket, inIface);
				break;
			default:
				break;
		}
	}

	// -----------------------------------------------------------------------
	// ARP handling
	// -----------------------------------------------------------------------
	private void handleArpPacket(Ethernet etherPacket, Iface inIface)
	{
		ARP arpPacket = (ARP) etherPacket.getPayload();
		int targetIp = ByteBuffer.wrap(
				arpPacket.getTargetProtocolAddress()).getInt();

		if (arpPacket.getOpCode() == ARP.OP_REQUEST)
		{
			// Reply only if the target IP matches this interface
			if (targetIp == inIface.getIpAddress())
			{ sendArpReply(etherPacket, inIface); }
		}
		else if (arpPacket.getOpCode() == ARP.OP_REPLY)
		{
			int senderIp = ByteBuffer.wrap(
					arpPacket.getSenderProtocolAddress()).getInt();
			MACAddress senderMac = MACAddress.valueOf(
					arpPacket.getSenderHardwareAddress());

			// Cache the learned mapping
			arpCache.insert(senderMac, senderIp);

			// Dequeue and forward any packets that were waiting
			ArpQueueEntry qEntry = arpQueues.remove(senderIp);
			if (qEntry != null)
			{
				if (qEntry.retryTimer != null)
				{ qEntry.retryTimer.cancel(); }
				for (Ethernet pkt : qEntry.packets)
				{
					pkt.setSourceMACAddress(qEntry.outIface.getMacAddress().toBytes());
					pkt.setDestinationMACAddress(senderMac.toBytes());
					sendPacket(pkt, qEntry.outIface);
				}
			}
		}
	}

	private void sendArpReply(Ethernet reqEther, Iface inIface)
	{
		ARP reqArp = (ARP) reqEther.getPayload();

		ARP reply = new ARP();
		reply.setHardwareType(ARP.HW_TYPE_ETHERNET);
		reply.setProtocolType(ARP.PROTO_TYPE_IP);
		reply.setHardwareAddressLength((byte) Ethernet.DATALAYER_ADDRESS_LENGTH);
		reply.setProtocolAddressLength((byte) 4);
		reply.setOpCode(ARP.OP_REPLY);
		reply.setSenderHardwareAddress(inIface.getMacAddress().toBytes());
		reply.setSenderProtocolAddress(IPv4.toIPv4AddressBytes(inIface.getIpAddress()));
		reply.setTargetHardwareAddress(reqArp.getSenderHardwareAddress());
		reply.setTargetProtocolAddress(reqArp.getSenderProtocolAddress());

		Ethernet replyEther = new Ethernet();
		replyEther.setEtherType(Ethernet.TYPE_ARP);
		replyEther.setSourceMACAddress(inIface.getMacAddress().toBytes());
		replyEther.setDestinationMACAddress(reqEther.getSourceMACAddress());
		replyEther.setPayload(reply);

		sendPacket(replyEther, inIface);
	}

	private void sendArpRequest(int targetIp, Iface outIface)
	{
		ARP req = new ARP();
		req.setHardwareType(ARP.HW_TYPE_ETHERNET);
		req.setProtocolType(ARP.PROTO_TYPE_IP);
		req.setHardwareAddressLength((byte) Ethernet.DATALAYER_ADDRESS_LENGTH);
		req.setProtocolAddressLength((byte) 4);
		req.setOpCode(ARP.OP_REQUEST);
		req.setSenderHardwareAddress(outIface.getMacAddress().toBytes());
		req.setSenderProtocolAddress(IPv4.toIPv4AddressBytes(outIface.getIpAddress()));
		req.setTargetHardwareAddress(new byte[]{0,0,0,0,0,0});
		req.setTargetProtocolAddress(IPv4.toIPv4AddressBytes(targetIp));

		Ethernet ether = new Ethernet();
		ether.setEtherType(Ethernet.TYPE_ARP);
		ether.setSourceMACAddress(outIface.getMacAddress().toBytes());
		ether.setDestinationMACAddress(BROADCAST_MAC);
		ether.setPayload(req);

		sendPacket(ether, outIface);
	}

	/**
	 * Queue a packet awaiting MAC resolution for nextHopIp and send an ARP request.
	 * Retries up to ARP_MAX_TRIES times; on failure, sends ICMP host unreachable.
	 */
	private void queueAndArp(final Ethernet etherPacket, final Iface inIface,
			final Iface outIface, final int nextHopIp)
	{
		ArpQueueEntry qEntry = arpQueues.computeIfAbsent(nextHopIp, k ->
		{
			ArpQueueEntry e = new ArpQueueEntry();
			e.outIface   = outIface;
			e.inIface    = inIface;
			e.nextHopIp  = nextHopIp;
			return e;
		});

		synchronized (qEntry)
		{
			qEntry.packets.add(etherPacket);
			if (qEntry.tries == 0)
			{
				sendArpRequest(nextHopIp, outIface);
				qEntry.tries = 1;
				scheduleArpRetry(nextHopIp);
			}
		}
	}

	private void scheduleArpRetry(final int nextHopIp)
	{
		ArpQueueEntry qEntry = arpQueues.get(nextHopIp);
		if (qEntry == null) { return; }

		Timer t = new Timer("ArpRetry-" + nextHopIp, true);
		qEntry.retryTimer = t;
		t.schedule(new TimerTask()
		{
			@Override
			public void run()
			{
				ArpQueueEntry entry = arpQueues.get(nextHopIp);
				if (entry == null) { return; }
				synchronized (entry)
				{
					if (entry.tries >= ARP_MAX_TRIES)
					{
						// Give up — send ICMP host unreachable for all queued packets
						arpQueues.remove(nextHopIp);
						for (Ethernet pkt : entry.packets)
						{ sendIcmpError(pkt, entry.inIface, (byte)3, (byte)1); }
					}
					else
					{
						sendArpRequest(nextHopIp, entry.outIface);
						entry.tries++;
						scheduleArpRetry(nextHopIp);
					}
				}
			}
		}, ARP_RETRY_MS);
	}

	// -----------------------------------------------------------------------
	// IPv4 handling
	// -----------------------------------------------------------------------
	private void handleIpPacket(Ethernet etherPacket, Iface inIface)
	{
		IPv4 ipPacket = (IPv4) etherPacket.getPayload();

		// Verify IP checksum
		short origCksum = ipPacket.getChecksum();
		ipPacket.resetChecksum();
		byte[] serial = ipPacket.serialize();
		ipPacket.deserialize(serial, 0, serial.length);
		if (origCksum != ipPacket.getChecksum())
		{
			System.out.println("Drop: bad IP checksum");
			return;
		}

		// RIP check: UDP destined to port 520
		if (ipPacket.getProtocol() == IPv4.PROTOCOL_UDP)
		{
			UDP udp = (UDP) ipPacket.getPayload();
			if (udp.getDestinationPort() == UDP.RIP_PORT)
			{
				handleRipPacket(etherPacket, inIface);
				return;
			}
		}

		// Is the destination one of our own interface IPs?
		for (Iface iface : this.interfaces.values())
		{
			if (ipPacket.getDestinationAddress() == iface.getIpAddress())
			{
				handleIpForUs(ipPacket, etherPacket, inIface);
				return;
			}
		}

		// Decrement TTL
		ipPacket.setTtl((byte)(ipPacket.getTtl() - 1));
		if (ipPacket.getTtl() <= 0)
		{
			sendIcmpError(etherPacket, inIface, (byte)11, (byte)0);  // time exceeded
			return;
		}
		ipPacket.resetChecksum();

		// Route lookup
		RouteEntry route = routeTable.lookup(ipPacket.getDestinationAddress());
		if (route == null)
		{
			sendIcmpError(etherPacket, inIface, (byte)3, (byte)0);  // net unreachable
			return;
		}

		Iface outIface = route.getInterface();

		/* Never route a packet back out the same interface it arrived on. */
		if (outIface == inIface) { return; }

		// Determine next-hop IP
		int nextHopIp = route.getGatewayAddress();
		if (nextHopIp == 0) { nextHopIp = ipPacket.getDestinationAddress(); }

		// ARP cache lookup
		ArpEntry arpEntry = arpCache.lookup(nextHopIp);
		if (arpEntry == null)
		{
			queueAndArp(etherPacket, inIface, outIface, nextHopIp);
			return;
		}

		// Forward
		etherPacket.setSourceMACAddress(outIface.getMacAddress().toBytes());
		etherPacket.setDestinationMACAddress(arpEntry.getMac().toBytes());
		sendPacket(etherPacket, outIface);
	}

	/** Handle an IP packet destined for one of our own interfaces. */
	private void handleIpForUs(IPv4 ipPacket, Ethernet etherPacket, Iface inIface)
	{
		byte proto = ipPacket.getProtocol();
		if (proto == IPv4.PROTOCOL_ICMP)
		{
			ICMP icmp = (ICMP) ipPacket.getPayload();
			if (icmp.getIcmpType() == 8)  // echo request
			{
				sendIcmpEchoReply(etherPacket, inIface);
				return;
			}
		}
		// TCP or UDP to a router address -> port unreachable
		if (proto == IPv4.PROTOCOL_TCP || proto == IPv4.PROTOCOL_UDP)
		{
			sendIcmpError(etherPacket, inIface, (byte)3, (byte)3);  // port unreachable
		}
	}

	// -----------------------------------------------------------------------
	// RIP packet handling
	// -----------------------------------------------------------------------
	private void handleRipPacket(Ethernet etherPacket, Iface inIface)
	{
		IPv4 ipPacket = (IPv4) etherPacket.getPayload();
		UDP  udp      = (UDP)  ipPacket.getPayload();
		RIPv2 rip     = (RIPv2) udp.getPayload();

		if (rip.getCommand() == RIPv2.COMMAND_REQUEST)
		{
			// Respond directly to the requester
			byte[] srcMac = etherPacket.getSourceMACAddress();
			sendRipResponseTo(inIface, ipPacket.getSourceAddress(), srcMac);
		}
		else if (rip.getCommand() == RIPv2.COMMAND_RESPONSE)
		{
			processRipResponse(rip, inIface, ipPacket.getSourceAddress());
		}
	}

	/**
	 * Update the route table based on an incoming RIP response.
	 * Applies Bellman-Ford: new_metric = advertised_metric + 1.
	 */
	private void processRipResponse(RIPv2 rip, Iface inIface, int srcIp)
	{
		for (RIPv2Entry entry : rip.getEntries())
		{
			int dstIp     = entry.getAddress() & entry.getSubnetMask();
			int maskIp    = entry.getSubnetMask();
			int newMetric = Math.min(entry.getMetric() + 1, RIP_INFINITY);

			RouteEntry existing = routeTable.find(dstIp, maskIp);

			// Never overwrite directly connected subnets
			if (existing != null && existing.getGatewayAddress() == 0)
			{ continue; }

			if (existing == null)
			{
				if (newMetric < RIP_INFINITY)
				{ routeTable.insert(dstIp, srcIp, maskIp, inIface, newMetric); }
			}
			else
			{
				// Update if better metric, or same gateway (refresh timeout)
				if (newMetric < existing.getMetric()
						|| existing.getGatewayAddress() == srcIp)
				{
					if (newMetric >= RIP_INFINITY)
					{
						routeTable.remove(dstIp, maskIp);
					}
					else
					{
						existing.setGatewayAddress(srcIp);
						existing.setInterface(inIface);
						existing.setMetric(newMetric);
						existing.updateTimestamp();
					}
				}
			}
		}
	}

	// -----------------------------------------------------------------------
	// ICMP helpers
	// -----------------------------------------------------------------------
	/** Send an ICMP echo reply in response to an echo request. */
	private void sendIcmpEchoReply(Ethernet reqEther, Iface inIface)
	{
		IPv4 reqIp   = (IPv4) reqEther.getPayload();
		ICMP reqIcmp = (ICMP) reqIp.getPayload();

		// Echo reply payload = identifier + sequence + data (everything after type/code/cksum)
		byte[] echoData = reqIcmp.getPayload().serialize();
		Data data = new Data(echoData);

		ICMP icmpReply = new ICMP();
		icmpReply.setIcmpType((byte) 0);   // echo reply
		icmpReply.setIcmpCode((byte) 0);
		icmpReply.setPayload(data);

		IPv4 ipReply = new IPv4();
		ipReply.setTtl((byte) 64);
		ipReply.setProtocol(IPv4.PROTOCOL_ICMP);
		ipReply.setSourceAddress(reqIp.getDestinationAddress());
		ipReply.setDestinationAddress(reqIp.getSourceAddress());
		ipReply.setPayload(icmpReply);

		routeAndSend(ipReply, inIface, reqIp.getSourceAddress(),
				reqEther.getSourceMACAddress());
	}

	/**
	 * Build and send an ICMP error message.
	 * @param origEther  the original packet that triggered the error
	 * @param inIface    interface the original packet arrived on
	 * @param type       ICMP type (3=dest unreachable, 11=time exceeded)
	 * @param code       ICMP code
	 */
	private void sendIcmpError(Ethernet origEther, Iface inIface,
			byte type, byte code)
	{
		IPv4 origIp = (IPv4) origEther.getPayload();

		// ICMP error payload: 4 unused bytes + original IP header + first 8 bytes of original payload
		byte[] origIpBytes  = origIp.serialize();
		int    headerBytes  = (origIp.getHeaderLength() & 0xF) * 4;
		int    payloadBytes = Math.min(8, origIpBytes.length - headerBytes);
		byte[] icmpPayload  = new byte[4 + headerBytes + payloadBytes];
		System.arraycopy(origIpBytes, 0, icmpPayload, 4, headerBytes + payloadBytes);

		ICMP icmp = new ICMP();
		icmp.setIcmpType(type);
		icmp.setIcmpCode(code);
		icmp.setPayload(new Data(icmpPayload));

		IPv4 ipErr = new IPv4();
		ipErr.setTtl((byte) 64);
		ipErr.setProtocol(IPv4.PROTOCOL_ICMP);
		ipErr.setSourceAddress(inIface.getIpAddress());
		ipErr.setDestinationAddress(origIp.getSourceAddress());
		ipErr.setPayload(icmp);

		routeAndSend(ipErr, inIface, origIp.getSourceAddress(),
				origEther.getSourceMACAddress());
	}

	/**
	 * Route an outbound IPv4 packet and send it.
	 * Falls back to using hintMac when the packet returns on the same interface.
	 */
	private void routeAndSend(IPv4 ipPacket, Iface inIface,
			int dstIp, byte[] hintMac)
	{
		RouteEntry route = routeTable.lookup(dstIp);
		if (route == null) { return; }

		Iface outIface  = route.getInterface();
		int nextHopIp   = route.getGatewayAddress();
		if (nextHopIp == 0) { nextHopIp = dstIp; }

		// Determine destination MAC
		byte[] dstMac;
		ArpEntry arpEntry = arpCache.lookup(nextHopIp);
		if (arpEntry != null)
		{
			dstMac = arpEntry.getMac().toBytes();
		}
		else if (outIface == inIface && hintMac != null)
		{
			// Response goes back same interface — use the sender's MAC directly
			dstMac = hintMac;
		}
		else
		{
			// No MAC available; send an ARP request and drop for now
			// (The original error-triggering packet is gone, so we can't queue it.)
			sendArpRequest(nextHopIp, outIface);
			return;
		}

		Ethernet ether = new Ethernet();
		ether.setEtherType(Ethernet.TYPE_IPv4);
		ether.setSourceMACAddress(outIface.getMacAddress().toBytes());
		ether.setDestinationMACAddress(dstMac);
		ether.setPayload(ipPacket);

		sendPacket(ether, outIface);
	}
}
