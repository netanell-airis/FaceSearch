import argparse
import logging
import os
import sys
from face_search import viz 

from PIL import Image
import numpy as np
import pandas as pd
import torch 
from face_search.search_index import SearchIndex
from face_search.utils import extract_signatures, get_files2process
from face_search.viz import render_query_res, serve_app
from face_search.fs_logger import logger_init


def filter_index_files(x):
    return  os.path.split(x)[-1] == 'video_summary.pth'


def run(args):
    imgs2proc = get_files2process(args.input_path)    
    # batch_transform(imgs2proc, transform=lambda x:rotate_90(x,clockwise=False))
    # return

    sigs = extract_signatures(imgs2proc)
    sigs = [np.array(x['signature']) for x in sigs]
    torch.save(dict(sigs= sigs, fnames=imgs2proc))

def img_path2info(fname):
    video_id = os.path.split(os.path.split(fname)[0])[1]
    video_id = video_id.split('.')[0]
    face_info = os.path.split(fname)[-1].split('_')
    face_id = int(face_info[1])
    frame_num = int(face_info[2].split('.')[0])
    return video_id, face_id, frame_num

def create_index(args):
    fnames = get_files2process(args.input_path)
    sigs = extract_signatures(fnames)
    # d = torch.load(os.path.join(args.input_path, 'sigs.pth'))
    # sigs = d['signatures']
    # fnames = d['fnames']
    embeddings = [np.array(x['embedding'])[np.newaxis,:] for x in sigs]
    embeddings = np.concatenate(embeddings, axis=0)
    face_info = [(x,)+img_path2info(x) for x in fnames]
    db = pd.DataFrame(face_info, columns = ['img_path','video_id','face_id','frame_num'])
    index_fname = args.save_path
    print(f'saving index into {index_fname}')
    torch.save(dict(embeddings=embeddings, db=db),index_fname)

def extract_query_sig(args):
    fnames = [args.query_img_path]
    sig = extract_signatures(fnames,detector_backend='retinaface')
    query = [np.array(x['embedding'])[np.newaxis,:] for x in sig]
    query = np.concatenate(query, axis=0)
    d = torch.load(args.input_path)
    embeddings= d['embeddings']
    db = d['db']
    num_face_id = 0
    for g1, db0 in db.groupby('video_id'):
        num_face_id += len(list(set(db0.face_id)))
    query = query / np.linalg.norm(query, axis=1)[:,np.newaxis]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:,np.newaxis]
    cdist = embeddings @ query.transpose()
    ix = np.argsort(-cdist, axis=0)[:,0]
    ix = ix[:10]
    db0 = db[ix]
    
def search_missing(missing_db_root, index_root, K=10):
    corpus = SearchIndex(512)
    corpus.corpus_dir = index_root 
    query = SearchIndex(512)
    query.corpus_dir = missing_db_root 
    search_results = list()
    with corpus as sig_tbl:
        enorm = corpus.t_tbl.enorm.values
        ns = enorm.size 
        tmplts = sig_tbl['templates'][:ns]
        ntmplts = tmplts / enorm[:,np.newaxis]        
        with query as qtbl:
            n0 = len(query.t_tbl)
            qnorm = query.t_tbl.enorm.values
            qtmplts = query.sigs['templates'][:n0]
            qntmplts = qtmplts / qnorm[:,np.newaxis]        
            # later, when we have large number of query 
            # and corpus templates, we can split the compute
            cos_sim = qntmplts @ ntmplts.transpose()
            cos_amax = np.argmax(cos_sim, axis=1)
            cos_max = cos_sim[np.arange(cos_sim.shape[0]),cos_amax]
        qidx = np.argsort(-cos_max)
        for i in range(5):
            qi = qidx[i]
            score  = cos_max[qi]
            ti = cos_amax[qi]
            q_face_id, q_video_id = query.t_tbl.loc[qi][['tid2fid','video_id']].astype(int)
            c_face_id, c_video_id = corpus.t_tbl.loc[ti][['tid2fid','video_id']].astype(int)            
            ftbl = query.face_tbl
            match_query = ftbl[(ftbl.video_id == q_video_id) &(ftbl.face_id == q_face_id)]
            ctbl = corpus.face_tbl
            t = ctbl[(ctbl.video_id == c_video_id) &(ctbl.face_id == c_face_id)]            
            frame_nums = t.frame_num.tolist()
            video_root = corpus.video_tbl.loc[c_video_id].video_fname
            fnames = [f'faceid_{c_face_id:04d}_{frame_num:04d}.png' for frame_num in frame_nums]
            fnames = [os.path.join(video_root,x) for x in fnames]
            name = match_query.person_id.tolist()[0] 
            res = dict()
            res['score'] = score 
            res['match_corpus'] = t 
            res['match_query'] = match_query 
            res['video'] = video_root
            search_results.append(res)
            logging.info(f'matching {name} to {c_face_id}={score:4f}')
            logging.info(f'cosine_sim={score}')
        
    # return cos_max_org, cos_amax, search_results
    return search_results 

def search_missing_usig_embeddings(missing_db_root, index_root, K=10):
    corpus = SearchIndex(512)
    corpus.corpus_dir = index_root 
    query = SearchIndex(512)
    query.corpus_dir = missing_db_root 

    with corpus as sig_tbl:
        enorm = corpus.face_tbl.enorm.values
        ns = enorm.size 
        nsigs = sig_tbl['embeddings'][:ns]
        # nsigs = sigs / enorm[:,np.newaxis]        
        file_name_list = list()
        with query as qtbl:
            n0 = len(query.face_tbl)
            cos_max = np.zeros((n0,K))
            cos_amax = np.zeros((n0,K),dtype=np.int32)
            qnorm = query.face_tbl.enorm.values
            for i, g in query.face_tbl.groupby(by='face_id'):
                ix = np.array(g.index)
                m0 = len(g)                
                qnsigs = qtbl['embeddings'][ix]
                # qnsigs = qsigs / qnorm[ix][:,np.newaxis]
                
                cos_sim = nsigs @ qnsigs.transpose()
                for k in range(K):
                    cos_max[ix,k] = cos_sim.max(axis=0)
                    cos_amax[ix,k] = cos_sim.argmax(axis=0)
                    cos_sim[cos_amax[ix,k],np.arange(m0)] = 0
                score = cos_max[ix,0].max()
                fid = int(g.iloc[0].face_id)
                logging.info(f'face_id={fid},num_faces={m0},max={score:.3f}')
        cos_max_org = cos_max.copy()
        for i in range(10):
            score = cos_max.max()
            # qi points to matching query
            qi, qj = np.where(cos_max==score)
            cos_max[qi,:] = 0
            match_query = query.face_tbl.loc[qi]
            # ti is pointing to matching template ()
            ti = cos_amax[qi,qj]
            video_id, face_id, frame_num, idx = (corpus.face_tbl.loc[ti,['video_id','face_id','frame_num','idx']].values).astype(np.int32).tolist()[0]
            video_fname = corpus.video_tbl.loc[video_id].video_fname
            tvid = corpus.face_tbl[corpus.face_tbl.video_id==video_id]
            if face_id >=0:
                t = tvid[tvid.face_id == face_id]
            else:
                t = tvid[(tvid.frame_num == frame_num) &(tvid.idx == idx)]
            for ix, r in t.iterrows():
                frame_num = int(r.frame_num)
                idx = int(r.idx)
                fname = f'face_{frame_num:05d}_{idx:04d}.png'
                fname = os.path.join(video_fname, fname)
                file_name_list.append(fname)
                name = match_query.person_id #.tolist()[0]
                logging.info(f'missing {name}')
                logging.info(f'{fname}')
                logging.info(f'cosine_sim={score}')
                if face_id < 0:
                    break
        with open('/tmp/files.txt','w') as fh:
            fh.writelines('\n'.join(file_name_list))
    return cos_max_org, cos_amax






def debug_sigs(args):
    from deepface import DeepFace
    from deepface.DeepFace import functions

    fnames = [args.query_img_path]
    d = torch.load(args.input_path)
    db = d['db']
    db0 = db[db.video_id.str.contains('33')]
    img_paths = db0.img_path.to_list()

    db_path = os.path.split(img_paths[-1])[0]
    db_path = os.path.split(db_path)[0]
    #args.query_img_path = img_paths[20]
    a = DeepFace.find(args.query_img_path, db_path, 
                      model_name='ArcFace', detector_backend='retinaface',
                      enforce_detection=False)


    sigs = extract_signatures(img_paths, detector_backend='retina_face')
    a = DeepFace.verify(img1_path = img_paths[0], img2_path=img_paths[1],
                    model_name='ArcFace',detector_backend='skip')

    sig = extract_signatures(fnames,detector_backend='retinaface')
    query = [np.array(x['embedding'])[np.newaxis,:] for x in sig]
    query = np.concatenate(query, axis=0)

def display_results():
    if 0:
        queries = [os.path.join('/tmp/query',x) for x in os.listdir('/tmp/query/')]
        q_list = list()
        for qname in queries:
            try:
                q = torch.load(qname)
            except:
                continue
            q_list += q
        scores = [x[1][0]['score'] for x in q_list]
        scores = np.array(scores)
        ix = np.argsort(scores)
        top_k_queries = [q_list[ix[i]] for i in range(10)]
    else:
        top_k_queries = torch.load('/tmp/best_queries.pth')

    layout = render_query_res(top_k_queries[:10])
    serve_app(layout)

def dump_html(results):
    headings = ['My Html']

    from yattag import Doc 
    doc, tag, text, line = Doc().ttl()
    with tag('table'):
        with tag('tr'):
            for x in headings:
                line('th', str(x))
        for res in results:
            q = res['match_query']
            v = res['match_corpus']        
            qimg = Image.open(q.iloc[0].fname)

            with tag('tr'):
                for x in row['match_corpus'].frame_num.tolist():
                    line('td', str(x))
        
    return doc

def plot_one_result(res):
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.rcParams['font.size'] = 18

    q = res['match_query']
    v = res['match_corpus']        
    qimg = Image.open(q.iloc[0].fname)
    face_id = int(v.face_id.tolist()[0])
    frame_nums = [int(x) for x in v.frame_num.tolist()]
    fnames = [f'faceid_{face_id:04d}_{fnum:04d}.png' for fnum in frame_nums]
    dimgs = [Image.open(os.path.join(res['video'], fname)) for fname in fnames]
    video = res['video']
    score = res['score']
    name = q.person_id.tolist()[0]
    n = int(len(dimgs) / 11 ) + 1 

    fig, axs = plt.subplots(nrows = n, ncols = 11,figsize=(15,15))
    fig.suptitle(f'{name}, score={score:.2f}', fontsize=16)
    for i,ax in enumerate(axs.flat):
        ax.axis('off')
        if i > len(dimgs):
            continue
        if i>0:
            ax.imshow(dimgs[i-1])
        else:
            ax.imshow(qimg)

    return fig 

def generate_image_tag(image_data):
    """Generates an HTML image tag from image data.

    Args:
    image_data: A byte string containing the image data.

    Returns:
    An HTML image tag.
    """
    import base64
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    image_tag = f'<img src="data:image/png;base64,{encoded_image}"/>'
    return image_tag

def create_html_table_with_images(results, res_dir):
    for i,res in enumerate(results):
        fig = plot_one_result(res)
        fig.savefig(f'/tmp/fig_{i:02d}.png')
    plot_one_result(results[0])
    doc = dump_html(results)
    with open('/tmp/test.html','w') as h:
        h.writelines(doc.getvalue())

    res = results[0]
    q = res['match_query']
    v = res['match_corpus']        
    qimg = Image.open(q.iloc[0].fname)
    face_id = int(v.face_id.tolist()[0])
    frame_nums = [int(x) for x in v.frame_num.tolist()]
    fnames = [f'faceid_{face_id:04d}_{fnum:04d}.png' for fnum in frame_nums]
    dimgs = [Image.open(os.path.join(res['video'], fname)) for fname in fnames]
    video = res['video']
    score = res['score']
    name = q.person_id 

                # Convert the PIL Image to a binary string
    image_bytes = qimg.tobytes()
                




    

    html = doc.render()





    #     for image in row:
    #   table_cell = html.Td(html.Img(src=image))
    #   table_row.append(table_cell)
    # table.append(table_row)

#   with open(output_html_file_path, "w") as f:
#     f.write(table.render())




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--corpus_dataset', type=str, default='video.mp4', help='root of index')  
    parser.add_argument('--query_dataset', type=str, default='output/', help='query') 
    args = parser.parse_args()
    #display_results()    
    logger_init()
    #import logging
    logger = logging.getLogger()

    root = os.environ['HOME']
    query_dataset = args.query_dataset 
    corpus_dataset = args.corpus_dataset
    video_dash_summary(corpus_dataset)
    # res_fname = os.path.split(query_dataset)

    # search_missing_usig_embeddings(query_dataset, corpus_dataset)
    search_results = search_missing(query_dataset, corpus_dataset)
    # search_results = torch.load('/tmp/search_results.pth')
    layout = viz.render_query_res(search_results)
    viz.serve_app(layout)

    # create_html_table_with_images(search_results, '/tmp/')
    sys.exit()
