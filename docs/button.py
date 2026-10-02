"""Sphinx directive that renders a download button."""

from typing import Any, ClassVar, NoReturn

import jinja2
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives import unchanged

BUTTON_TEMPLATE = jinja2.Template(
    """
<a href="{{ link }}" download>
    <div class="button" type="button">{{ text }}</div>
</a>
"""
)


# placeholder node for document graph
class ButtonNode(nodes.General, nodes.Element):
    """Placeholder node for a button in the document graph."""


class ButtonDirective(Directive):
    """Directive that inserts a download button with a ``text`` and a ``link`` option."""

    required_arguments = 0

    option_spec: ClassVar[dict[str, Any]] = {
        "text": unchanged,
        "link": unchanged,
    }

    # this will execute when your directive is encountered
    # it will insert a button_node into the document that will
    # get visited during the build phase
    def run(self) -> list[nodes.Node]:
        """Create the button node."""
        env = self.state.document.settings.env
        app = env.app

        app.add_css_file("button.css")

        node = ButtonNode()
        node["text"] = self.options["text"]
        node["link"] = self.options["link"]
        return [node]


# build phase visitor emits HTML to append to output
def html_visit_button_node(self: Any, node: ButtonNode) -> NoReturn:
    """Render the button node as HTML."""
    html = BUTTON_TEMPLATE.render(text=node["text"], link=node["link"])

    self.body.append(html)
    raise nodes.SkipNode


# if you want to be pedantic, define text, latex, manpage visitors too..


def setup(app: Any) -> dict[str, Any]:
    """Register the button node and directive with Sphinx."""
    app.add_node(ButtonNode, html=(html_visit_button_node, None))
    app.add_directive("button", ButtonDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
