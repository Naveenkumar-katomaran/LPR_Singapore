import pika
import json
import os

RABBITMQ_HOST = '127.0.0.1'
QUEUE_NAME = 'vehicle_entries'

def callback(ch, method, properties, body):
    try:
        data = json.loads(body)
        print("\n" + "="*50)
        print(f" [x] Received Event: {data.get('event_id')}")
        print(f"     Plate: {data.get('number_plate')}")
        print(f"     Type:  {data.get('vehicle_type')}")
        print(f"     Time:  {data.get('detected_at')}")
        print(f"     Plate Image:   {data.get('number_plate_image')}")
        print(f"     Vehicle image: {data.get('vehicle_image_url')}")
        print("="*50)
        
        # Acknowledge the message so it's removed from queue
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"Error processing message: {e}")

def main():
    print(f" [*] Connecting to RabbitMQ at {RABBITMQ_HOST}...")
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()

        channel.queue_declare(queue=QUEUE_NAME, durable=False)
        
        print(f" [*] Waiting for messages in '{QUEUE_NAME}'. To exit press CTRL+C")
        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

        channel.start_consuming()
    except KeyboardInterrupt:
        print("\nStopping consumer...")
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == '__main__':
    main()
