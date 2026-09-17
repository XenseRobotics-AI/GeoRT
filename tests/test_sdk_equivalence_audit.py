import unittest

from scripts.audit_wuji_sdk_equivalence import extract_asset


class EmbeddedAssetTests(unittest.TestCase):
    def test_extracts_only_unique_parseable_right_hand2(self):
        right = b'<robot name="right"><joint name="r_pinky_mcp_flex"/></robot>'
        data = b'\x00<robot name="bad">\x00' + right + b'<robot name="left"/>garbage'
        self.assertEqual(extract_asset(data),right)

    def test_missing_or_multiple_assets_fail_explicitly(self):
        right = b'<robot name="right"><joint name="r_pinky_mcp_flex"/></robot>'
        for data in (b'garbage',b'<robot name="other"></robot>',right+right):
            with self.assertRaisesRegex(ValueError,'missing or ambiguous'):extract_asset(data)


if __name__ == '__main__': unittest.main()
