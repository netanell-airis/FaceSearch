import os
import io
import dash
import dash_table as dt
# import dash_core_components as dcc
import dash_html_components as html
# import plotly.express as px
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
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    src = get_img_src(img)
    html_img = html.Img(src=src, height=f'{size:d}px')
    cap = html.P(caption)
    div = html.Div((html_img, cap))
    td = html.Td(div)
    return td


def img_tbl_layout(img_list, captions,img_width=100,num_columns=10):
    table_rows = list()
    cells = list()
    for img_dir,cap in zip(img_list, captions):
        cells.append(get_html_cell(img_dir, cap))
        if len(cells) % num_columns == 0:
            table_rows.append(html.Tr(cells))
            cells = list()
    if len(cells):
        table_rows.append(html.Tr(cells))
    return html.Table(table_rows)

def video_summary_layout(ftbl,video_root, group_by_faceid=False):
    face_layouts = list()
    all_imgs = list()
    all_captions = list()
    for fid,g in ftbl.groupby(by='face_id'):
        fid = int(fid)
        frames = g.frame_num.astype(int).tolist()
        if fid >=0:
            img_list = [os.path.join(video_root,f'faceid_{fid:04d}_{fn:04d}.png') for fn in frames]
            captions = [f'{fid},{fr}' for fr in frames]
            layout = img_tbl_layout(img_list, captions)
            face_layouts.append(layout)
        else:
            idx = g.idx.astype(int).tolist()
            captions = [f'-1,{fn}' for fn,ix in zip(frames, idx)]
            img_list = [os.path.join(video_root, f'face_{fn:05d}_{ix:04d}.png') for fn,ix in zip(frames, idx)]
            face_layouts.append(img_tbl_layout(img_list, captions))
        all_imgs += img_list 
        all_captions += captions

    h1 = html.H1(os.path.split(video_root)[-1])
    if group_by_faceid:
        layout = html.Div(face_layouts)
    else:
        layout = img_tbl_layout(all_imgs,all_captions)
    
    layout = html.Div([h1, layout])
    return layout






def render_query_res(query_results):
    html_rows = list()
    html_tables = list()
    scores = list()
    for tbls in query_results:
        score = tbls['score']
        scores.append(score)
        query = tbls['match_query']
        t = tbls['match_corpus']
        video_root = tbls['video']
        person_id = query.person_id.iloc[0]
        fnames = query.fname.tolist()
        fname = fnames[0]
        caption = f'name={person_id},score={score:.3f}'

        cells = [get_html_cell(fname,caption,300)]
        cells = cells + [get_html_cell(fname,'',300) for fname in fnames[1:]]
        frames = t.frame_num.astype(int).tolist()
        html_tables.append(html.Table([html.Tr(cells)]))
        cells = list()
        face_id = int(t.face_id.iloc[0])
        labels = [f'{face_id:04d}_{fn:04d}' for fn in frames]
        fnames = [os.path.join(tbls['video'], f'faceid_{lbl}.png') for lbl in labels]
        cn = 0
        html_rows = list()
        for img_dir, caption in zip(fnames, labels):
            if len(cells) %10 == 0 and len(cells):
                html_rows.append(html.Tr(cells))
                cells = list()
            img = Image.open(img_dir)
            cells.append(get_html_cell(img, caption,size=128))
            cn += 1
        
        if len(cells):
            html_rows.append(html.Tr(cells))
        html_tables.append(html.Table(html_rows))        
    # table = html.Table(html_rows)
    s0 = scores[0]
    tbls = list()
    for ix,score in enumerate(scores):
        tbls.append(html.H1(f'Query score {score:.2f}'))
        tbls.append(html_tables[2*ix])
        tbls.append(html_tables[2*ix+1])

    layout = html.Div(tbls)
    
    return layout 


def serve_app(layout, port = 8080):
    # Create a Dash app
    app = dash.Dash('serve_app')
    # Set the app layout
    app.layout = layout
    app.run_server(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    # import plotly.io as pio
    # pio.write_html(layout, file='/tmp/table11.html')


