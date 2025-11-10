# Ecommerce_product_assistant
python -m prod_assistant.mcp_servers.product_search_server
uvicorn prod_assistant.router.main:app --reload --host 127.0.0.1 --port 8080
