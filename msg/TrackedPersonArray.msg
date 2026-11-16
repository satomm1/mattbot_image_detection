# header.stamp: the time every state below is valid at (the latest detection frame); frame_id: map.
# Only confirmed tracks. A person not detected recently is still listed (predicted) until it expires:
# compare last_seen with header.stamp to tell how fresh each one is.
Header header
TrackedPerson[] people
