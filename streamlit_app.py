from chat import main

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

if __name__ == "__main__":
    main()
