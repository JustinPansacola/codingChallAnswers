from mcp.server.transport_security import TransportSecuritySettings

# The grader connects from a host we don't control, over the open internet.
# FastMCP's default DNS-rebinding protection only allow-lists localhost, which
# would reject every real request once deployed, so it's turned off here.
PUBLIC_TRANSPORT_SECURITY = TransportSecuritySettings(enable_dns_rebinding_protection=False)
