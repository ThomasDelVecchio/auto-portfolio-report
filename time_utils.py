from datetime import datetime

# -------------------------- Timezone Helper --------------------------

try:
    # Python 3.9+ standard library
    from zoneinfo import ZoneInfo

    def get_eastern_now():
        """Return current time in America/New_York (Eastern Time)."""
        return datetime.now(ZoneInfo("America/New_York"))

except Exception:
    # Fallback if zoneinfo isn't available; use pytz if present
    try:
        import pytz
        ET = pytz.timezone("US/Eastern")

        def get_eastern_now():
            """Return current time in US/Eastern using pytz."""
            return datetime.now(ET)

    except Exception:
        # Last-resort fallback: system local time
        def get_eastern_now():
            """Fallback to system local time if timezone libraries not available."""
            return datetime.now()
