"""Dash application to explore final clustering images produced by the project."""
from __future__ import annotations

import argparse
import base64
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

try:  # Pillow is optional and only used for metadata if available.
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _discover_images(image_dir: Path) -> List[Path]:
    image_dir = image_dir.expanduser().resolve()
    if not image_dir.exists():
        return []
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def _build_options(records: Iterable[Path], image_dir: Path) -> List[dict]:
    options = []
    for path in records:
        try:
            label = str(path.relative_to(image_dir))
        except ValueError:
            label = str(path.name)
        options.append({"label": label, "value": str(path)})
    return options


def _load_image(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    mime = "image/tiff" if path.suffix.lower() in {".tif", ".tiff"} else "image/png"
    return f"data:{mime};base64,{encoded}"


def _format_bytes(size: int) -> str:
    step = 1024.0
    units = ["bytes", "KiB", "MiB", "GiB"]
    for unit in units:
        if size < step or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "bytes" else f"{size} {unit}"
        size /= step
    return f"{size:.1f} GiB"


def _image_metadata(path: Path, image_dir: Path) -> List[html.Li]:
    stats = path.stat()
    metadata = [
        html.Li([html.Strong("File:"), f" {path.name}"]),
    ]
    try:
        rel_path = path.relative_to(image_dir)
    except ValueError:
        rel_path = path
    metadata.append(html.Li([html.Strong("Relative path:"), f" {rel_path}"]))
    metadata.append(
        html.Li([html.Strong("Size:"), f" {_format_bytes(stats.st_size)}"])
    )
    metadata.append(
        html.Li(
            [
                html.Strong("Last modified:"),
                " ",
                datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M"),
            ]
        )
    )
    if Image is not None:
        try:
            with Image.open(path) as img:
                metadata.append(
                    html.Li(
                        [
                            html.Strong("Resolution:"),
                            f" {img.width} x {img.height} px",
                        ]
                    )
                )
        except Exception:
            pass
    return metadata


def create_dash_app(image_dir: Path, *, title: Optional[str] = None) -> Dash:
    image_dir = image_dir.expanduser().resolve()
    records = _discover_images(image_dir)
    options = _build_options(records, image_dir)

    app = Dash(__name__)
    app.title = title or "Final Image Dashboard"

    initial_value = options[0]["value"] if options else None

    description = textwrap.dedent(
        f"""
        Use the controls below to explore the final clustering images produced by the
        analysis pipeline. Populate the directory `{image_dir}` with exported images
        (PNG, JPEG, or TIFF) and use the filter box to quickly find specific files.
        Adjust the width slider to change the on-screen size of the preview.
        """
    ).strip()

    app.layout = html.Div(
        [
            html.H1(app.title),
            dcc.Markdown(description),
            html.Div(
                [
                    html.Label("Filter images by name"),
                    dcc.Input(
                        id="filter-text",
                        type="text",
                        placeholder="Type to filter...",
                        debounce=True,
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            dcc.Dropdown(
                id="image-dropdown",
                options=options,
                value=initial_value,
                placeholder="No images found" if not options else None,
            ),
            html.Div(
                [
                    html.Label("Preview width (pixels)"),
                    dcc.Slider(
                        id="width-slider",
                        min=200,
                        max=1600,
                        value=800,
                        step=50,
                        marks={200: "200", 800: "800", 1600: "1600"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                style={"margin": "1.5rem 0"},
            ),
            html.Div(id="image-metadata"),
            html.Div(id="image-preview", style={"marginTop": "1rem"}),
            dcc.Store(
                id="image-store",
                data={"options": options, "directory": str(image_dir)},
            ),
        ],
        style={"maxWidth": "960px", "margin": "0 auto", "padding": "2rem 1rem"},
    )

    @app.callback(
        Output("image-dropdown", "options"),
        Output("image-dropdown", "value"),
        Input("filter-text", "value"),
        State("image-store", "data"),
        State("image-dropdown", "value"),
    )
    def _update_dropdown(filter_text: Optional[str], store_data: dict, current_value: Optional[str]):
        all_options = store_data["options"]
        if not all_options:
            return all_options, None

        if filter_text:
            normalized = filter_text.lower()
            filtered = [
                option
                for option in all_options
                if normalized in option["label"].lower()
            ]
        else:
            filtered = all_options

        if not filtered:
            return [], None

        values = {option["value"] for option in filtered}
        if current_value in values:
            value = current_value
        else:
            value = filtered[0]["value"]
        return filtered, value

    @app.callback(
        Output("image-preview", "children"),
        Output("image-metadata", "children"),
        Input("image-dropdown", "value"),
        Input("width-slider", "value"),
        State("image-store", "data"),
    )
    def _render_image(selected_path: Optional[str], width: int, store_data: dict):
        if not selected_path:
            message = html.Div(
                [
                    html.P("No images available."),
                    html.P(
                        "Add PNG, JPEG, or TIFF files to the configured directory and refresh the app.",
                        style={"color": "#555"},
                    ),
                ]
            )
            return message, html.Ul([])

        path = Path(selected_path)
        if not path.exists():
            raise PreventUpdate

        try:
            image_src = _load_image(path)
        except Exception as exc:  # pragma: no cover - defensive
            error = html.Div(
                [
                    html.P("Failed to load image"),
                    html.Pre(str(exc)),
                ],
                style={"color": "red"},
            )
            return error, html.Ul([])

        metadata = html.Ul(_image_metadata(path, Path(store_data["directory"])))
        preview = html.Img(
            src=image_src,
            style={"width": f"{int(width)}px", "border": "1px solid #ccc", "padding": "0.5rem"},
        )
        return preview, metadata

    return app


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the final image dashboard.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("final_outputs"),
        help="Directory containing the exported final images (default: ./final_outputs).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8050, help="Port to serve the dashboard (default: 8050).")
    parser.add_argument(
        "--title",
        default="Final Image Dashboard",
        help="Custom title for the dashboard window and header.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    app = create_dash_app(args.image_dir, title=args.title)
    app.run_server(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":  # pragma: no cover
    main()
