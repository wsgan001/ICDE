from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
import ttk
import pymysql as mdb
import os

class App(object):
	def __init__(self, window,ip,user,pswd,db):
		self.ip=ip
		self.user=user
		self.pswd=pswd
		self.db=db

		window.wm_title("ICDE")
		window.geometry('550x250')

		conn = mdb.connect(ip,user,pswd,db)
		cursor = conn.cursor()

		#set the current row to zero
		self.current_row=0

		#search bar
		self.sb_text  = StringVar()
		self.sb_entry = Entry(window, textvariable=self.sb_text,width=29)
		self.sb_entry.grid(row=self.current_row,column=2)
		self.sb_btn = Button(window,text="Search",width=5)
		self.sb_btn.grid(row=self.current_row,column=3)

		#update button
		self.message = Label(text="", fg = "red").grid(row=self.current_row+1,column=3)
		self.update_btn=Button(window,text="Update Selected",width=12,command=self.update)
		self.update_btn.grid(row=self.current_row,column=4)
		self.current_row += 1

		#tree view
		self.tree = ttk.Treeview(window)
		self.tree["columns"] = ("one","two", "three", "four", "five")
		self.tree.column("one",width=100)
		self.tree.column("two",width=100)
		self.tree.column("three",width=100)
		self.tree.column("four",width=100)
		self.tree.column("five",width=100)
		self.tree.column("#0",width=30,anchor='w')

		self.tree.grid(row=self.current_row, column=0, columnspan=6)
		self.tree.heading("#0",text="Id", anchor='w')
		self.tree.heading("one",text="Firstname")
		self.tree.heading("two",text="Middlename")
		self.tree.heading("three",text="Lastname")
		self.tree.heading("four",text="ID Type")
		self.tree.heading("five",text="Validity Date")

		for i in range(0,6):
			cursor.execute("""SELECT * FROM user""")
			
		for row in cursor:
			self.tree.insert('','end',text=row[0],values=(row[1],row[2],row[3],row[4],row[5]))
				
		#scrollbar
		self.scroll= Scrollbar(window)
		self.scroll.grid(row=self.current_row,column=7,rowspan=10)
		self.tree.configure(yscrollcommand=self.scroll.set)
		self.scroll.configure(command=self.tree.yview)
		self.current_row += 1

	def update(self):
		user_id = self.tree.item(self.tree.selection())["text"]
		fname = self.tree.item(self.tree.selection())["values"][0]
		mname = self.tree.item(self.tree.selection())["values"][1]
		lname = self.tree.item(self.tree.selection())["values"][2]
		id_type = self.tree.item(self.tree.selection())["values"][3]
		valid_date = self.tree.item(self.tree.selection())["values"][4]

		self.update_wind = Toplevel()
		self.update_wind.title("Update ID")
		Label(self.update_wind, text = "Firstname").grid(row=0, column=1)
		Entry(self.update_wind, textvariable=StringVar(self.update_wind, value = fname), state = "readonly").grid(row=0,column=2)
		Label(self.update_wind, text = "Middlename").grid(row=1, column=1)
		Entry(self.update_wind, textvariable=StringVar(self.update_wind, value = mname), state = "readonly").grid(row=1,column=2)
		Label(self.update_wind, text = "Lastname").grid(row=2, column=1)
		Entry(self.update_wind, textvariable=StringVar(self.update_wind, value = lname), state = "readonly").grid(row=2,column=2)
		Label(self.update_wind, text = "ID Type").grid(row=3, column=1)
		Entry(self.update_wind, textvariable=StringVar(self.update_wind, value = id_type), state = "readonly").grid(row=3,column=2)
		Label(self.update_wind, text = "Valid Until").grid(row=4, column=1)
		Entry(self.update_wind, textvariable=StringVar(self.update_wind, value = valid_date), state = "readonly").grid(row=4,column=2)
		
		self.update_btn1=Button(self.update_wind,text="Update ID",width=12,command=self.upload_id)
		self.update_btn1.grid(row=5,column=2)

		self.update_wind.mainloop()

	def upload_id(self):
		self.update_wind.withdraw()
		self.upload_wind = Toplevel()
		self.upload_wind.title("Upload ID")
		Label(self.upload_wind, text = "File").grid(row=1, column=0)
		#self.filename_holder = Entry(self.upload_wind, textvariable=StringVar()).grid(row=2,column=0)
		self.browse_btn=Button(self.upload_wind,text="Browse",width=6,command=self.load_file)
		self.browse_btn.grid(row=1,column=2)
		self.upload_wind.mainloop()

	def load_file(self):
		self.upload_wind.withdraw()
		self.filename = tkFileDialog.askopenfilename(initialdir = "tf_files/source_images/",title = "Browse file",filetypes = (("all files","*.*"),("jpg","*.jpg*"),("jpeg","*.jpeg*"),("png","*.png*")))
		filen = self.filename.split("/")
		path = filen[8]+"/"+filen[9]
		print(path)
		#run classifier
		print("\n==========Classifying ID Type...==========\n")
		os.system('python -m scripts.label_image --graph=tf_files/retrained_graph.pb  --image=tf_files/{0}'.format(path))
		print("\n========Extracting Name and Date...========\n")
		#run extractor
		classified = open('tf_files/classified.txt','r').read()
		id_type = classified
		os.system('python scripts/data_extractor.py --image tf_files/{0} --type {1}'.format(path,id_type))
		#run validator
		os.system('python scripts/validate.py')
		#validate();
		return

	def validate(self):
		user_id = self.tree.item(self.tree.selection())["text"]
		fname = self.tree.item(self.tree.selection())["values"][0]
		mname = self.tree.item(self.tree.selection())["values"][1]
		lname = self.tree.item(self.tree.selection())["values"][2]

		#validate from db
		#select card type from identification card
		#assign to id_type
		#insert to user 

		#select expiry_date from identification card
		#assign to valid_date
		#insert to user 


def main():
	window= Tk()
	start= App(window,"localhost","root","Nutella4898","icde")
	window.mainloop()

if __name__ == "__main__":
	main()