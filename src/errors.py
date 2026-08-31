class UpstreamServiceError(RuntimeError):
    """A required external dependency could not complete the request."""


def raise_gemini_error(error):
    if "API_KEY_IP_ADDRESS_BLOCKED" in str(error):
        message = "The Gemini API key does not allow this server's IP address. Update its allowed-IP list."
    else:
        message = "Gemini is unavailable. Check the API key, quota, and model access, then retry."
    raise UpstreamServiceError(message) from error
