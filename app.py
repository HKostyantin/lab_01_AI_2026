import base64
import io

import numpy as np
from PIL import Image
from dash import Dash, dcc, html, Input, Output, State

import core

FOLDER = "train_images"
app = Dash(__name__)
app.title = "Lab 1 - Adaptive Recognition System"

METHODS = list(core.METRICS) + ["Ортогональні області"]
clf = core.Classifier()
training = core.load_training(FOLDER)  
samples = core.list_samples(FOLDER)    

def fmt(vec, norm=False):
    arr = np.round(vec, 4 if norm else 0).reshape(core.ROWS, core.COLS)
    return html.Table([html.Tr([html.Td(f"{v:g}", style={"border": "1px solid #ccc", "padding": "3px 8px"})
                                for v in row]) for row in arr])


def stats_table():
    rows = [html.Tr([html.Th(h) for h in ("Клас", "Зразків", "Центр", "min", "max")])]
    for l, c in clf.clusters.items():
        rows.append(html.Tr([html.Td(l), html.Td(c["n"])] +
                            [html.Td(", ".join(f"{v:.3f}" for v in c[k])) for k in ("center", "min", "max")]))
    return html.Table(rows, style={"fontSize": "11px"})


def sample_card(path, label):
    src = "data:image/bmp;base64," + base64.b64encode(open(path, "rb").read()).decode()
    abs_v, norm_v = core.extract_features(Image.open(path))
    return html.Div([html.B(f"Клас {label}: {path.split('/')[-1]}"), html.Br(),
                     html.Img(src=src, style={"width": "100px", "border": "1px solid #999"}),
                     html.Div(["abs: ", ", ".join(f"{v:g}" for v in abs_v)], style={"fontSize": "11px"}),
                     html.Div(["norm: ", ", ".join(f"{v:.3f}" for v in norm_v)], style={"fontSize": "11px"})],
                    style={"display": "inline-block", "width": "260px", "verticalAlign": "top", "margin": "6px"})


def samples_view(mode):
    cards = [sample_card(p, l) for l, ps in samples.items() for p in (ps[:1] if mode == "etalon" else ps)]
    return html.Details([html.Summary("Еталони" if mode == "etalon" else "Навчальна послідовність"), *cards],
                        open=mode == "etalon")


def img_from(contents):
    return Image.open(io.BytesIO(base64.b64decode(contents.split(",", 1)[1])))


app.layout = html.Div([
    html.H1("Lab 1 - Adaptive Recognition System"),
    html.Div([
        html.Label("Режим: "),
        dcc.RadioItems(id="mode", value="train", inline=True, options=[
            {"label": "Навчання (центри кластерів)", "value": "train"},
            {"label": "Еталони (1 зразок на клас)", "value": "etalon"}]),
        html.Label("Метод / норма: ", style={"marginTop": "8px", "display": "block"}),
        dcc.Dropdown(id="method", options=METHODS, value=METHODS[0], clearable=False, style={"width": "260px"}),
    ]),
    html.H3("Кластери (статистика навчальної вибірки)"),
    html.Div(id="clusters"),
    html.Div(id="samples"),
    html.H3("Невідомий образ"),
    dcc.Upload(id="upload", children=html.Button("Завантажити BMP"), accept=".bmp,.png,.jpg,.jpeg"),
    html.Div(id="output", style={"marginTop": "15px"}),
], style={"maxWidth": "900px", "margin": "30px auto", "fontFamily": "Arial, sans-serif"})


@app.callback(Output("clusters", "children"), Output("samples", "children"), Input("mode", "value"))
def set_mode(mode):
    clf.train(training) if mode == "train" else clf.etalons(training)
    return stats_table(), samples_view(mode)


@app.callback(Output("output", "children"), Input("upload", "contents"), Input("method", "value"),
              Input("mode", "value"), State("upload", "filename"))
def recognize(contents, method, mode, filename):
    if contents is None:
        return "Завантажте зображення."
    try:
        img = img_from(contents)
        abs_vec, norm_vec = core.extract_features(img)
        if abs_vec.sum() == 0:
            return html.Div("Порожнє зображення.", style={"color": "red"})
        clf.train(training) if mode == "train" else clf.etalons(training)
        winner, scores = clf.predict(norm_vec, method)
    except Exception as e:
        return html.Div(f"Помилка: {e}", style={"color": "red"})
    return html.Div([
        html.P(f"Файл: {filename}"),
        html.Img(src=contents, style={"maxWidth": "240px", "border": "1px solid #999"}),
        html.H4("Абсолютний вектор ознак (4x5)"), fmt(abs_vec),
        html.H4("Нормований вектор ознак"), fmt(norm_vec, norm=True),
        html.H4("Міри відповідності по класах"),
        html.Ul([html.Li(f"Клас {l}: {s}", style={"fontWeight": "bold" if l == winner else "normal"})
                 for l, s in scores.items()]),
        html.H3(f"Результат: клас {winner}", style={"color": "green"}),
    ])


if __name__ == "__main__":
    app.run(debug=True)
