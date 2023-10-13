import io
import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from PIL import Image
import base64



# Convert the PIL Image to a base64 string

def get_img_src(img):
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img64 = base64.b64encode(buffer.getvalue())
    src = 'data:image/png;base64,{}'.format(img64.decode('utf-8'))
    return src

def get_html_cell(img, caption='',size=100):
    src = get_img_src(img)
    html_img = html.Img(src=src, height=f'{size:d}px')
    cap = html.P(caption)
    div = html.Div((html_img, cap))
    td = html.Td(div)
    return td

def render_query_res(query_results):
    html_rows = list()
    for qinfo, qres in query_results:
        fname = qinfo["fname"]
        caption = f'fname={fname[-20:]}'
        cells = list()
        cells.append(get_html_cell(qinfo['img'],caption,300))
        for res in qres:
            score = res['score']
            caption = f'score={score:.3f}'
            cells.append(get_html_cell(res['img'],caption))

        html_rows.append(html.Tr(cells))
    table = html.Table(html_rows)
    layout = html.Div([
        html.H1('Query results'),
        table,
    ])
    return layout 


def serve_app(layout):
    # Create a Dash app
    app = dash.Dash('serve_app')
    # Set the app layout
    app.layout = layout
    app.run_server(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
    # import plotly.io as pio
    # pio.write_html(layout, file='/tmp/table11.html')


