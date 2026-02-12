"""End-to-end tests for the web dashboard using Playwright."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Page

# Mark all tests as browser tests
pytestmark = pytest.mark.browser


class TestDashboardLoading:
    """Tests for dashboard initial loading."""

    def test_dashboard_loads(self, dashboard_page: Page) -> None:
        """Dashboard page loads and displays title."""
        assert "Crow's Nest" in dashboard_page.title()
        # Check sidebar header
        header = dashboard_page.locator(".sidebar-header h1")
        assert header.text_content() == "Crow's Nest"

    def test_overview_is_default_view(self, dashboard_page: Page) -> None:
        """Overview view is shown by default."""
        # Overview nav item should be active
        nav_item = dashboard_page.locator('.nav-item[data-view="overview"]')
        class_attr = nav_item.get_attribute("class") or ""
        assert "active" in class_attr

        # Overview section should be visible
        overview = dashboard_page.locator("#overview-view")
        assert overview.is_visible()

    def test_stats_cards_displayed(self, dashboard_page: Page) -> None:
        """Statistics cards are displayed in overview."""
        stat_cards = dashboard_page.locator(".stat-card")
        assert stat_cards.count() >= 6

        # Check specific stat labels exist (use more specific selectors)
        assert dashboard_page.locator(".stat-card:has-text('Active Thunks')").is_visible()
        assert dashboard_page.locator(".stat-card:has-text('Completed')").is_visible()
        assert dashboard_page.locator(".stat-card:has-text('Failed')").is_visible()


class TestNavigation:
    """Tests for sidebar navigation."""

    def test_navigate_to_thunks(self, dashboard_page: Page) -> None:
        """Clicking Thunks nav item shows thunks view."""
        # Click on Thunks nav item
        dashboard_page.locator('.nav-item[data-view="thunks"]').click()

        # Wait for view to be visible
        dashboard_page.wait_for_selector("#thunks-view.active", state="visible")

        # Verify thunks view is shown
        assert dashboard_page.locator("#thunks-view").is_visible()
        assert not dashboard_page.locator("#overview-view").is_visible()

        # Nav item should be active
        nav_item = dashboard_page.locator('.nav-item[data-view="thunks"]')
        class_attr = nav_item.get_attribute("class") or ""
        assert "active" in class_attr

    def test_navigate_to_agents(self, dashboard_page: Page) -> None:
        """Clicking Agents nav item shows agents view."""
        dashboard_page.locator('.nav-item[data-view="agents"]').click()
        dashboard_page.wait_for_selector("#agents-view.active", state="visible")

        assert dashboard_page.locator("#agents-view").is_visible()

    def test_navigate_to_queues(self, dashboard_page: Page) -> None:
        """Clicking Queues nav item shows queues view."""
        dashboard_page.locator('.nav-item[data-view="queues"]').click()
        dashboard_page.wait_for_selector("#queues-view.active", state="visible")

        assert dashboard_page.locator("#queues-view").is_visible()

    def test_navigate_to_events(self, dashboard_page: Page) -> None:
        """Clicking Events nav item shows events view."""
        dashboard_page.locator('.nav-item[data-view="events"]').click()
        dashboard_page.wait_for_selector("#events-view.active", state="visible")

        assert dashboard_page.locator("#events-view").is_visible()

    def test_navigate_to_config(self, dashboard_page: Page) -> None:
        """Clicking Config nav item shows config view."""
        dashboard_page.locator('.nav-item[data-view="config"]').click()
        dashboard_page.wait_for_selector("#config-view.active", state="visible")

        assert dashboard_page.locator("#config-view").is_visible()

    def test_navigate_back_to_overview(self, dashboard_page: Page) -> None:
        """Can navigate back to overview from another view."""
        # Navigate to thunks
        dashboard_page.locator('.nav-item[data-view="thunks"]').click()
        dashboard_page.wait_for_selector("#thunks-view.active", state="visible")

        # Navigate back to overview
        dashboard_page.locator('.nav-item[data-view="overview"]').click()
        dashboard_page.wait_for_selector("#overview-view.active", state="visible")

        assert dashboard_page.locator("#overview-view").is_visible()


class TestThunksView:
    """Tests for the thunks view functionality."""

    def test_thunks_list_view(self, dashboard_page: Page) -> None:
        """Thunks list view displays correctly."""
        dashboard_page.locator('.nav-item[data-view="thunks"]').click()
        dashboard_page.wait_for_selector("#thunks-view.active", state="visible")

        # List tab should be active by default
        list_tab = dashboard_page.locator('.tab-btn[data-tab="list"]')
        class_attr = list_tab.get_attribute("class") or ""
        assert "active" in class_attr

        # List view should be visible
        assert dashboard_page.locator("#thunk-list-view").is_visible()

    def test_thunks_dag_view(self, dashboard_page: Page) -> None:
        """Thunks DAG view can be switched to."""
        dashboard_page.locator('.nav-item[data-view="thunks"]').click()
        dashboard_page.wait_for_selector("#thunks-view.active", state="visible")

        # Click DAG tab
        dashboard_page.locator('.tab-btn[data-tab="dag"]').click()

        # Wait for DAG view to become active
        dashboard_page.wait_for_selector("#thunk-dag-view.active", state="visible")

        # DAG container should be visible
        assert dashboard_page.locator("#thunk-dag-container").is_visible()

        # DAG toolbar buttons should exist
        assert dashboard_page.locator("text=Fit to View").is_visible()
        assert dashboard_page.locator("text=Reset Zoom").is_visible()

    def test_thunk_filter_dropdown(self, dashboard_page: Page) -> None:
        """Thunk status filter dropdown works."""
        dashboard_page.locator('.nav-item[data-view="thunks"]').click()
        dashboard_page.wait_for_selector("#thunks-view.active", state="visible")

        # Check filter dropdown exists and has options
        filter_select = dashboard_page.locator("#thunk-filter")
        assert filter_select.is_visible()

        # Check options
        options = filter_select.locator("option")
        assert options.count() >= 4

    def test_refresh_button(self, dashboard_page: Page) -> None:
        """Refresh button is clickable."""
        dashboard_page.locator('.nav-item[data-view="thunks"]').click()
        dashboard_page.wait_for_selector("#thunks-view.active", state="visible")

        # Find and click refresh button
        refresh_btn = dashboard_page.locator("#thunks-view .btn-refresh")
        assert refresh_btn.is_visible()
        refresh_btn.click()


class TestConnectionStatus:
    """Tests for WebSocket connection status indicator."""

    def test_connection_status_visible(self, dashboard_page: Page) -> None:
        """Connection status indicator is visible in sidebar."""
        status = dashboard_page.locator("#connectionStatus")
        assert status.is_visible()

    def test_connection_status_has_indicator(self, dashboard_page: Page) -> None:
        """Connection status has a dot indicator."""
        dot = dashboard_page.locator("#connectionStatus .status-dot")
        assert dot.is_visible()


class TestEventsView:
    """Tests for the events view functionality."""

    def test_events_filter_input(self, dashboard_page: Page) -> None:
        """Events filter input is usable."""
        dashboard_page.locator('.nav-item[data-view="events"]').click()
        dashboard_page.wait_for_selector("#events-view.active", state="visible")

        filter_input = dashboard_page.locator("#event-filter")
        assert filter_input.is_visible()

        # Type in filter
        filter_input.fill("test")
        assert filter_input.input_value() == "test"

    def test_clear_events_button(self, dashboard_page: Page) -> None:
        """Clear events button is visible and clickable."""
        dashboard_page.locator('.nav-item[data-view="events"]').click()
        dashboard_page.wait_for_selector("#events-view.active", state="visible")

        clear_btn = dashboard_page.locator(".btn-clear")
        assert clear_btn.is_visible()
        clear_btn.click()


class TestResponsiveness:
    """Tests for responsive design elements."""

    def test_sidebar_is_present(self, dashboard_page: Page) -> None:
        """Sidebar navigation is always present."""
        assert dashboard_page.locator(".sidebar").is_visible()
        assert dashboard_page.locator(".nav-menu").is_visible()

    def test_main_content_area(self, dashboard_page: Page) -> None:
        """Main content area is visible."""
        assert dashboard_page.locator(".content").is_visible()


class TestAgentsView:
    """Tests for the agents view functionality."""

    def test_agents_view_elements(self, dashboard_page: Page) -> None:
        """Agents view has expected elements."""
        dashboard_page.locator('.nav-item[data-view="agents"]').click()
        dashboard_page.wait_for_selector("#agents-view.active", state="visible")

        # Check for show terminated checkbox
        checkbox = dashboard_page.locator("#show-terminated")
        assert checkbox.is_visible()

        # Check for agent tree container
        tree = dashboard_page.locator("#agent-tree")
        assert tree.is_visible()

        # Check for detail panel
        detail = dashboard_page.locator("#agent-detail")
        assert detail.is_visible()
