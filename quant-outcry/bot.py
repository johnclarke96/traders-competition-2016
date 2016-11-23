from model import ridge_model_comp
from tradersbot import TradersBot
import numpy as np

hostname = 'mangocore.pw'
username = 'jaclarke@mit.edu'
password = 'about-large-things'
t = TradersBot(hostname, username, password)

ridge = ridge_model_comp()
news = []
def updatePredict(msg, order):
	newsBody = msg["news"]["body"]
	newsBody = str(newsBody).split(' ')
	vals = [float(item.replace(';','')) for item in newsBody if ';' in item]
	vals.append(float(newsBody[-1]))  
	news.append(vals)
	print 'Data ', news

	if len(news) >= 2:
		x = []
		d1 = np.array(news[-2])
		d0 = np.delete(np.array(news[-1]),[2])
		x = np.append(x, d0)
		x = np.append(x,d1)
		print 'Prediction: ', ridge.predict([x])[0]

t.onNews = updatePredict

def hi(msg,order):
	print "hey"

t.onAckRegister = hi
# ######################################################
# # each time AAPL trade happens for $x, make bid
# # and ask at $x-0.02, $x+0.02, respectively
# def marketMake(msg, order):
#     for trade in msg["trades"]:
#         if trade["ticker"] == 'AAPL':
#             px = trade["price"]
#             order.addBuy('AAPL', 10, px - 0.02)
#             order.addSell('AAPL', 10, px + 0.02)

#t.onTrade = marketMake





t.run()