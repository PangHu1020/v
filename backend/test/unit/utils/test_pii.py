"""Unit tests for ``backend.v.utils.pii``."""

from __future__ import annotations

from backend.v.utils.pii import contains_pii, mask_pii


class TestMaskPhone:
    def test_masks_mainland_mobile(self) -> None:
        out = mask_pii("客户电话是 13812341234")
        assert "13812341234" not in out
        assert "138" in out and "1234" in out  # head/tail kept
        assert "*" in out

    def test_phone_inside_longer_digit_run_untouched(self) -> None:
        # An order id that happens to contain 11 digits should not be masked
        # as a phone (boundary guards).
        out = mask_pii("订单号 201381234123499")
        assert "201381234123499" in out


class TestMaskIdCard:
    def test_masks_18_digit_id(self) -> None:
        out = mask_pii("身份证 11010519900307123X")
        assert "11010519900307123X" not in out
        assert out.startswith("身份证 1101")
        assert out.endswith("123X")

    def test_id_not_tagged_as_bank_card(self) -> None:
        # 18-char id (17 digits + X) keeps head=4 tail=4, the id-card mask,
        # not the bank-card mask (head=0 tail=4).
        out = mask_pii("11010519900307123X")
        assert out[:4] == "1101"


class TestMaskBankCard:
    def test_masks_16_digit_card(self) -> None:
        out = mask_pii("卡号 6222021234567890")
        assert "6222021234567890" not in out
        assert out.endswith("7890")
        # bank card keeps no head
        assert out.startswith("卡号 *")


class TestMaskEmail:
    def test_masks_local_part(self) -> None:
        out = mask_pii("邮箱 zhangsan@example.com")
        assert "zhangsan@" not in out
        assert "@example.com" in out


class TestMaskAddress:
    def test_replaces_full_address(self) -> None:
        out = mask_pii("收货地址：浙江省杭州市西湖区文一西路100号5栋301室")
        assert "文一西路" not in out
        assert "【已脱敏地址】" in out

    def test_ordinary_prose_untouched(self) -> None:
        # No administrative-token + building-token anchor → not an address.
        text = "客户说想退货"
        assert mask_pii(text) == text


class TestMixedAndEdge:
    def test_multiple_pii_in_one_sentence(self) -> None:
        out = mask_pii("客户 13812341234 要求寄到北京市朝阳区建国路88号2单元")
        assert "13812341234" not in out
        assert "建国路" not in out

    def test_empty_string(self) -> None:
        assert mask_pii("") == ""

    def test_no_pii_returns_unchanged(self) -> None:
        text = "客户对订单 SO123 的物流速度不满意"
        assert mask_pii(text) == text


class TestContainsPii:
    def test_true_on_phone(self) -> None:
        assert contains_pii("打 13800001111") is True

    def test_false_on_clean(self) -> None:
        assert contains_pii("客户偏好顺丰") is False

    def test_false_on_empty(self) -> None:
        assert contains_pii("") is False
