from admin_sessions import AdminSessionStore, parse_admin_session_timeout


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_session_expiry_touch_and_revoke():
    clock = FakeClock()
    sessions = AdminSessionStore(300, clock=clock)
    token = sessions.create()
    assert sessions.validate(token)

    clock.advance(299)
    assert sessions.touch(token)
    clock.advance(299)
    assert sessions.validate(token)
    clock.advance(2)
    assert not sessions.validate(token)

    replacement = sessions.create()
    assert sessions.revoke(replacement)
    assert not sessions.validate(replacement)


def test_timeout_configuration_bounds():
    assert parse_admin_session_timeout(None) == 300
    assert parse_admin_session_timeout("600") == 600
    assert parse_admin_session_timeout("0") == 300
    assert parse_admin_session_timeout("86401") == 300
    assert parse_admin_session_timeout("invalid") == 300
