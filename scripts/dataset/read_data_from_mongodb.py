"""A script to read docs (title, content, title-definition, title-synonms) from mongodb and output json encoded docs."""

import pymongo as pm
import json


# 获取连接
client = pm.MongoClient('localhost', 27017)
# 连接数据库
db = client.test
# 获取集合
stb = db.mba
# 获取数据信息
datas = stb.find()

mba_docs = {}
for idx, data in enumerate(datas):
    # title -> (definition, content, synonyms)
    mba_docs[data['main']] = (data['def'] if 'def' in data else 'empty',
                              data['raw'],
                              data['key'] if 'key' in data else 'empty')

output_path = '../../data/mba/raw/mba_def.json'
with open(output_path, 'w', encoding='utf8') as file:
    json.dump(mba_docs, file)